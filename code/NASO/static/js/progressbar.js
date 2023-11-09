function updateProgress (progressUrl) {
    fetch(progressUrl).then(function(response) {
        if (response.ok) {
            response.json().then(function(data) {
                // update the appropriate UI components
                
                setProgress(data.state, data.details);
                if (data.state === "PROGRESS") {
                    setTimeout(updateProgress, 1000, progressUrl);
                }
                else if (!"state" in data) {
                    setTimeout(updateProgress, 1000, progressUrl);
                }
                if(data.state === 'SUCCESS') {
                    // display a toast message training is finished
                    toastr.success("Training beendet", 'INFO');
                }
            });
        } else {
            throw new Error('Something went wrong');
        }
    })
    .catch((error) => {
        console.log(error);
        setTimeout(updateProgress, 2000, progressUrl);
    });
}


function setProgress(state, details) {
    try {
        const progressBars = [...document.getElementsByClassName("training-progress-bar")];
        let width = (details.current / details.total) * 100;
        progressBars.forEach(element => {
            element.children[1].children[0].setAttribute("value",width);
            // element.children[1].children[0].style.width = width + "%";
    
            element.children[0].innerText = "Epoche " + details.current + " von " + details.total;
        });
        
        if (details.metrics.loss) {
            $('#training-tags').empty();
            console.log(details);
            $('#training-tags').append(
                '<div class="tag is-light is-info ">Curent loss: ' + details.metrics.loss + '</div>'
            );
        }

        let progressLink = '/runs/' + details.run_id + '/details/';
        if (details.autokeras) {
            progressLink = '/runs/autokeras/' + details.run_id + '/details/';
        }
        const link = document.getElementById('training_detail_link')
        if (link) {
            link.href = progressLink;
        }
        
    } catch {}
    
}

updateProgress(progressUrl);
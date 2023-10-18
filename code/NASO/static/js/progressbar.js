function updateProgress (progressUrl) {
    fetch(progressUrl).then(function(response) {
        response.json().then(function(data) {
            // update the appropriate UI components
            
            setProgress(data.state, data.details);
            if (data.state === "PROGRESS") {
                setTimeout(updateProgress, 1000, progressUrl);
            }
            if (!"state" in data) {
                setTimeout(updateProgress, 1000, progressUrl);
            }
            if(data.state === 'SUCCESS') {
                // display a toast message training is finished
                toastr.success("Training beendet", 'INFO');
            }
        });
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
        $('#training-tags').empty();
        console.log(details);
        $('#training-tags').append(
            '<div class="tag is-light is-info ">Curent accuracy: ' + details.metrics.accuracy + '</div>'
        );

        const progressLink = '/runs/' + details.run_id + '/details/';
        const link = document.getElementById('training_detail_link')
        if (link) {
            link.href = progressLink;
        }
        
    } catch {}
    
}

updateProgress(progressUrl);
function updateProgress (progressUrl, idAppendix) {
    fetch(progressUrl).then(function(response) {
        if (response.ok) {
            response.json().then(function(data) {
                // update the appropriate UI components
                
                setProgress(data.state, data.details, idAppendix);
                if (data.state === "PROGRESS") {
                    setTimeout(updateProgress, 1000, progressUrl, idAppendix);
                }
                else if (!"state" in data) {
                    setTimeout(updateProgress, 1000, progressUrl, idAppendix);
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
        setTimeout(updateProgress, 2000, progressUrl, idAppendix);
    });
}


function setProgress(state, details, idAppendix) {
    try {
        $('#task_progress_'+idAppendix).attr("value", (details.current / details.total) * 100);
        $('#import_job_state_text_'+idAppendix).text("Epoche " + details.current + " von " + details.total);
        if (details.metrics.loss) {
            $('#training-tags-'+idAppendix).empty();
            console.log(details);
            $('#training-tags-'+idAppendix).append(
                '<div class="tag is-light is-info ">Curent loss: ' + details.metrics.loss + '</div>'
            );
        }

        let progressLink = '/runs/' + details.run_id + '/details/';
        if (details.autokeras) {
            progressLink = '/runs/autokeras/' + details.run_id + '/details/';
        }
        const link = document.getElementById('training_detail_link_'+idAppendix)
        if (link) {
            link.href = progressLink;
        }
        
    } catch {}
    
}
for (const progress in progressUrls) {
    updateProgress(progressUrls[progress], progress);
}
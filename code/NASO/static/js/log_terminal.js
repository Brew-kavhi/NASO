let newMessages = 0;
let lastPosition = 0;

document.addEventListener('DOMContentLoaded', function () {
    const showLogButton = document.getElementById('showLogButton');

    let timerId = undefined;
    if (showLogButton) {
        timerId = setInterval(updateLog, 15000);
    }
    showLogButton.addEventListener('click', function () {
        
        $('#logContainer').toggleClass('d-none');
        if (showLogButton.innerText ==='Show Log') {
            showLogButton.innerText = 'Hide Log';
            
            clearInterval(timerId);
            timerId = setInterval(updateLog, 1000);
        } else {
            showLogButton.innerText = 'Show Log';
            
            clearInterval(timerId);
            timerId = setInterval(updateLog, 5000);
        }
    });
});

function updateLog() {
    fetch(logUrl + `?last_position=${lastPosition}`)
    .then(response => response.json())
    .then(data => {
        const logContainer = document.getElementById('logContainer');
        const showLogButton = document.getElementById('showLogButton');
        if (data.new_position != lastPosition) {
            logContainer.innerHTML = (data.log_content) + logContainer.innerHTML;
            if (showLogButton.innerText === 'Show Log') {
                newMessages++;
                $('#new_logs_counter').text(newMessages);
                $('#new_logs_counter').removeClass('d-none');
                lastPosition = data.new_position;
            } else {
                newMessages = 0;
                $('#new_logs_counter').addClass('d-none');
                lastPosition = data.new_position;
            }
        }
    });
}

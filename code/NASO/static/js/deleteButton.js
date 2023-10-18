
$(document).ready(function() {
    // When a delete button is clicked, update the modal content
    $('.delete-btn').on('click', function() {
        $($(this).data('target')).addClass('is-active');
        var itemID = $(this).data('item-id');
        $('#itemToDeleteID').text(itemID);
        $('#delete_action_url').val($(this).data('action'));
    });

    // When the "Delete" button in the modal is clicked, handle the deletion
    $('#deleteConfirmBtn').on('click', function() {
        var deleteURL = $('#delete_action_url').val();
        
        $.ajax({
            type: 'GET',
            url: deleteURL,
            success: function(data) {
                // Handle success, e.g., show a message or reload the page
                $('#accordion'+data['id']).remove();
                toastr.success("Datensatzx geloscht", 'INFO');
                closeDeleteModal();
            },
            error: function() {
                // Handle error
                toastr.error("Datensatz konnte nicht geloscht werden", 'FEHLER');
            }
        });
    });
});

function  closeDeleteModal() {
    $('#confirmDeleteModal').removeClass('is-active');
}
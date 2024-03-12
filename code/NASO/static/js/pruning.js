function togglePruning() {
    $('#div_id_pruning_method').parent().toggleClass('d-none');
    $('#div_id_pruning_scheduler').parent().toggleClass('d-none');
    $('#div_id_pruning_policy').parent().toggleClass('d-none');
}

function toggleClustering() {
    $('#div_id_number_of_clusters').parent().toggleClass('d-none');
    $('#div_id_centroids_init').parent().toggleClass('d-none');
}

function showPruning() {
    $('#div_id_pruning_method').parent().removeClass('d-none');
    $('#div_id_pruning_scheduler').parent().removeClass('d-none');
    $('#div_id_pruning_policy').parent().removeClass('d-none');
}

function handlePruningMethodChange(selectElement) {
    var selectedPruningMethodId = $(selectElement).val();
    var selectedPruningMethod = pruningMethodOptions.find(function(opt) {
        return opt.id == selectedPruningMethodId;
    });

    $('#pruning-methods-arguments').empty();
    $('#pruning-methods-arguments').removeClass('p-4 mb-3');

    if (!selectedPruningMethod) {
        return ;
    }
    showPruning();

    // Clear existing input fields

    // Display input fields for each argument
    for (const argumentName of selectedPruningMethod.jsonConfig) {
        $('#pruning-methods-arguments').append(addInputField(argumentName, 'pruning-method_argument_'));  // Append input fields
        $('#pruning-methods-arguments').addClass('p-4 mb-3');
    }
}

function handlePruningSchedulerChange(selectElement) {
    var selectedPruningSchedulerId = $(selectElement).val();
    var selectedPruningScheduler = pruningSchedulerOptions.find(function(opt) {
        return opt.id == selectedPruningSchedulerId;
    });

    $('#pruning-scheduler-arguments').empty();
    $('#pruning-scheduler-arguments').removeClass('p-4 mb-3');

    if (!selectedPruningScheduler) {
        return ;
    }

    // Clear existing input fields

    // Display input fields for each argument
    for (const argumentName of selectedPruningScheduler.jsonConfig) {
        $('#pruning-scheduler-arguments').append(addInputField(argumentName, 'pruning-scheduler_argument_'));  // Append input fields
        $('#pruning-scheduler-arguments').addClass('p-4 mb-3');
    }
}

function handlePruningPolicyChange(selectElement) {
    var selectedPruningPolicyId = $(selectElement).val();
    var selectedPruningPolicy = pruningPolicyOptions.find(function(opt) {
        return opt.id == selectedPruningPolicyId;
    });

    $('#pruning-policy-arguments').empty();
    $('#pruning-policy-arguments').removeClass('p-4 mb-3');

    if (!selectedPruningPolicy) {
        return ;
    }

    // Clear existing input fields

    // Display input fields for each argument
    for (const argumentName of selectedPruningPolicy.jsonConfig) {
        $('#pruning-policy-arguments').append(addInputField(argumentName, 'pruning-policy_argument_'));  // Append input fields
        $('#pruning-policy-arguments').addClass('p-4 mb-3');
    }
}

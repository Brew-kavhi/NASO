function handleOptimizerChange(selectElement) {
    var selectedOptimizerId = $(selectElement).val();
    var selectedOptimizer = optimizerOptions.find(function(opt) {
        return opt.id == selectedOptimizerId;
    });

    $('#optimizer-arguments').empty();
    $('#optimizer-arguments').removeClass('p-4 mb-3');
    
    if (!selectedOptimizer) {
        return ;
    }
    
    // Clear existing input fields

    // Display input fields for each argument
    for (const argumentName of selectedOptimizer.jsonConfig) {
        $('#optimizer-arguments').append(addInputField(argumentName,'optimizer_argument_'));  // Append input fields
        $('#optimizer-arguments').addClass('p-4 mb-3');
    }
}

function handleLossChange(selectElement) {
    var selectedLossId = $(selectElement).val();
    var selectedLoss = lossOptions.find(function(opt) {
        return opt.id == selectedLossId;
    });

    $('#loss-arguments').empty();
    $('#loss-arguments').removeClass('p-4 mb-3');
    
    if (!selectedLoss) {
        return ;
    }
    
    // Clear existing input fields

    // Display input fields for each argument
    for (const argumentName of selectedLoss.jsonConfig) {
        $('#loss-arguments').append(addInputField(argumentName, 'loss_argument_'));  // Append input fields
        $('#loss-arguments').addClass('p-4 mb-3');
    }
}

function handleLayerChange(selectElement) {
    var selectedLayerId = $(selectElement).val();
    var selectedLayer = layerOptions.find(function(opt) {
        return opt.id == selectedLayerId;
    });

    $('#layer-arguments').empty();
    $('#layer-arguments').removeClass('p-4 mb-3');
    
    if (!selectedLayer) {
        return ;
    }
    
    // Clear existing input fields

    // Display input fields for each argument
    for (const argumentName of selectedLayer.jsonConfig) {
        $('#layer-arguments').append(addInputField(argumentName, 'layer_argument_'));  // Append input fields
        $('#layer-arguments').addClass('p-4 mb-3');
    }
}


function handleMetricChange(selectElement) {
    var selectedMetricIds = $(selectElement).val();
    var selectedMetrics = [];

    selectedMetricIds.forEach(function(metricId) {
        var metric = metricOptions.find(function(m) {
            return m.id == metricId;
        });
        if (metric) {
            selectedMetrics.push(metric);
        }
    });

    var metricsArguments = {};

    // Collect arguments from selected metrics
    selectedMetrics.forEach(function(metric) {
        Object.assign(metricsArguments, metric.jsonConfig);
    });

    // Clear existing input fields
    $('#metrics-arguments').empty();
    $('#metrics-arguments').removeClass('p-4 mb-3');
    let metric_weight = $('#metric_weights_arguments');
    if (metric_weight) {
        metric_weight.empty();
        metric_weight.removeClass('p-5');
    }

    if (selectedMetrics.length === 0) {
        return ;
    }

    for (const selectedMetric of selectedMetrics) {
        if (selectedMetric.jsonConfig.length > 0) {
                
            let metricContainerHTML = '<div class="card shadow-none m-3 w-auto" style="min-width:33%; max-width: 50%"> <div class="card-header">' + selectedMetric.name + '</div><div class="card-body">';
            
            for (const argumentName of selectedMetric.jsonConfig) {
                metricContainerHTML += addInputField(argumentName, 'metric_argument_' + selectedMetric.id + '_');
            }
            
            metricContainerHTML += '</div></div>';
            $('#metrics-arguments').append(metricContainerHTML);  // Append input fields
            $('#metrics-arguments').addClass('p-4 mb-3');
            
            if (metric_weight) {
                weight_html = '<div class="mb-4 row"><label class="col-form-label col-lg-4">' + selectedMetric.name + '</label>' +
                '<div class="col-lg-8"><input class="form-control" type="text" name="metric_weight_' + selectedMetric.id + '" value="1.0"></div></div>';
                metric_weight.append(weight_html);
                metric_weight.addClass('p-5');
            }
        }
    }
    if (metric_weight) {
        weight_html = '<div class="mb-4 row"><label class="col-form-label col-lg-4">Model Size</label>' +
        '<div class="col-lg-8"><input class="form-control" type="text" name="metric_weight_modelsize" value="1.0"></div></div>' + 
        '<div class="mb-4 row"><label class="col-form-label col-lg-4">Execution Time</label>' +
        '<div class="col-lg-8"><input class="form-control" type="text" name="metric_weight_executiontime" value="1.0"></div></div>';
        metric_weight.append(weight_html);
        metric_weight.addClass('p-5');
    }
}

function handleCallbackChange(selectElement) {
    var selectedCallbackIds = $(selectElement).val();
    var selectedCallbacks = [];

    selectedCallbackIds.forEach(function(callbackId) {
        var callback = callbackOptions.find(function(m) {
            return m.id == callbackId;
        });
        if (callback) {
            selectedCallbacks.push(callback);
        }
    });

    var callbacksArguments = {};

    // Collect arguments from selected Callbacks
    selectedCallbacks.forEach(function(Callback) {
        Object.assign(callbacksArguments, Callback.jsonConfig);
    });

    // Clear existing input fields
    $('#callbacks-arguments').empty();
    $('#callbacks-arguments').removeClass('p-4 mb-3');
   

    if (selectedCallbacks.length === 0) {
        return ;
    }

    for (const selectedCallback of selectedCallbacks) {
        if (selectedCallback.jsonConfig.length > 0) {
                
            let callbackContainerHTML = '<div class="card shadow-none m-3 w-auto" style="min-width:33%; max-width: 50%"> <div class="card-header">' + selectedCallback.name + '</div><div class="card-body">';
            
            for (const argumentName of selectedCallback.jsonConfig) {
                callbackContainerHTML += addInputField(argumentName, 'callback_argument_' + selectedCallback.id + '_');
            }
            
            callbackContainerHTML += '</div></div>';
            $('#callbacks-arguments').append(callbackContainerHTML);  // Append input fields
            $('#callbacks-arguments').addClass('p-4 mb-3');
        }
    }
}

function handleKerasTunerChange(selectElement) {
    var selectedTunerId = $(selectElement).val();
    var selectedTuner = kerasTunerOptions.find(function(opt) {
        return opt.id == selectedTunerId;
    });

    $('#tuner-arguments').empty();
    $('#tuner-arguments').removeClass('p-4 mb-3');
    
    if (!selectedTuner) {
        return ;
    }
    
    // Clear existing input fields

    // Display input fields for each argument
    for (const argumentName of selectedTuner.jsonConfig) {
        $('#tuner-arguments').append(addInputField(argumentName, 'tuner_argument_'));  // Append input fields
        $('#tuner-arguments').addClass('p-4 mb-3');
    }
}


function handleKerasBlockChange(selectElement) {
    var selectedBlockId = $(selectElement).val();
    var selectedBlock = kerasBlocksOptions.find(function(opt) {
        return opt.id == selectedBlockId;
    });

    $('#layer-arguments').empty();
    $('#layer-arguments').removeClass('p-4 mb-3');
    
    if (!selectedBlock) {
        return ;
    }
    
    // Clear existing input fields

    // Display input fields for each argument
    for (const argumentName of selectedBlock.jsonConfig) {
        $('#layer-arguments').append(addInputField(argumentName, 'layer_argument_'));  // Append input fields
        $('#layer-arguments').addClass('p-4 mb-3');
    }
}

function handleDatasetLoaderChange(selectElement) {
    var selectedDatasetLoaderId = $(selectElement).val();
    // fetch the datasets for this loader from runs/get_datasets/<dataset_loader_id>:
    $.ajax({
        url: '/runs/get_dataset/' + selectedDatasetLoaderId,
        type: 'GET',
        success: function(data) {
            $('#id_dataset').autocomplete({
                source: data,
                minLength:0,
                autoFocus:true,
            })
        },
        error: function(error) {
            console.log(error);
        }
    });
}

function addInputField(argument, prefix) {
    if ('name' in argument) {
        var defaultValue = argument['default'];
        label = argument['name'];
        inputName = prefix + argument['name']
        return '<div class="mb-4 row"><label class="col-form-label col-lg-2">' + label + '</label>' +
        '<div class="col-lg-10 d-flex"><input class="form-control" type="text" name="' + inputName + '" value="' + 
        defaultValue + '"><span class="icon"><i class="fa fa-help icon" data-tooltip="Type: ' + argument['dtype'] + '">?</i></span></div></div>';
    }
    return '';
    
}
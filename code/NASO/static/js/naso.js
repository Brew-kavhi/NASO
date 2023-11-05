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
    

    if (selectedMetrics.length === 0) {
        return ;
    }
    metricObjectiveOptions = new Map();

    for (const selectedMetric of selectedMetrics) {
        if (selectedMetric.jsonConfig.length > 0) {
                
            let metricContainerHTML = '<div class="card shadow-none m-3 w-auto" style="min-width:33%; max-width: 50%"> <div class="card-header">' + selectedMetric.name + '</div><div class="card-body">';
            
            for (const argumentName of selectedMetric.jsonConfig) {
                metricContainerHTML += addInputField(argumentName, 'metric_argument_' + selectedMetric.id + '_');
                
            }
            
            metricContainerHTML += '</div></div>';
            $('#metrics-arguments').append(metricContainerHTML);  // Append input fields
            $('#metrics-arguments').addClass('p-4 mb-3');
            for (const argumentName of selectedMetric.jsonConfig) {
                if (argumentName['name'] == 'name') {
                    metricObjectiveOptions.set(selectedMetric.id, $('#metric_argument_' + selectedMetric.id + '_name').val());
                    //add onchange listener to the objective field
                    $('#metric_argument_' + selectedMetric.id + '_name').attr('metric_id', selectedMetric.id);
                    $('#metric_argument_' + selectedMetric.id + '_name').change(function() {    
                        console.log('metricchange');
                        metricObjectiveOptions.set(Number($(this).attr('metric_id')), $(this).val());
                        renderMetricWeights();
                    });
                    break;
                }
            }
            
            
        }
    }
    renderMetricWeights();
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
    callbackObjectiveOptions = [];

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
        if (selectedCallback.metrics && selectedCallback.metrics.length > 0) {
            for (const metric of selectedCallback.metrics) {
                if (!callbackObjectiveOptions.includes(metric)) {
                    console.log(metric);
                    callbackObjectiveOptions.push(metric);
                }
            }
        }
    }
    renderMetricWeights();
}

function renderMetricWeights() {
    // render a card with fields to set the weights for each metric in the ObjectiveOptions
    let metric_weight = $('#metric_weights_arguments');
    if (metric_weight.length > 0) {
        metric_weight.empty();
        metric_weight.removeClass('p-5');
        
        $('#id_objective').autocomplete({
            source: fixedObjectiveOptions.concat(callbackObjectiveOptions).concat(Array.from(metricObjectiveOptions.values())),
            minLength: 0,
            select: function(event, ui) {
                $('#id_objective').val(ui.item.value);
                if (ui.item.value ==='metrics') {
                    $('#metric_weights_arguments').addClass('d-block');
                    $('#metric_weights_arguments').removeClass('d-none');
                }
                else {
                    $('#metric_weights_arguments').removeClass('d-block');
                    $('#metric_weights_arguments').addClass('d-none');
                }
                return false;
            }
        }).focus(function() {
            $(this).autocomplete('search', $(this).val())
        });
        
        let metricContainerHTML = '<div class="card shadow-none m-3 w-auto"> <div>Weights</div><div class="card-body d-flex flex-wrap">';
        for (const metric of fixedObjectiveOptions.concat(callbackObjectiveOptions).concat(Array.from(metricObjectiveOptions.values()))) {  
            value = 1.0;
            if (metricWeights.has(metric)) {
                value = metricWeights.get(metric);
            }  
            else {
                metricWeights.set(metric, value);
            }
            if (metric !== 'metrics') {
                weight_html = '<div class="col-4 mb-4 row"><label class="col-form-label col-lg-4">' + metric + '</label>' +
                '<div class="col-lg-8"><input class="form-control" type="text" name="metric_weight_' + metric + '" value="' + value + '"></div></div>';
                metricContainerHTML += weight_html;
            }
            
        }
        metricContainerHTML += '</div></div>';
        metric_weight.append(metricContainerHTML);  // Append input fields
        metric_weight.addClass('p-4 mb-3');
        for (const metric of fixedObjectiveOptions.concat(callbackObjectiveOptions).concat(Array.from(metricObjectiveOptions.values()))) {  
            if (metric !== 'metrics') {
                $('#metric_weights_arguments input[name="metric_weight_' + metric + '"]').change(function() {
                    metricName = $(this).attr('name').slice(14);
                    metricWeights.set(metricName, $(this).val());
                });
            }            
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
        '<div class="col-lg-10 d-flex"><input class="form-control" type="text" id="' + inputName+ '" name="' + inputName + '" value="' + 
        defaultValue + '"><span class="icon"><i class="fa fa-help icon" data-tooltip="Type: ' + argument['dtype'] + '">?</i></span></div></div>';
    }
    return '';   
}

let metricWeights = new Map();
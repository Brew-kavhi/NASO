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
        if ('name' in argumentName) {
            var defaultValue = argumentName['default'];
            var inputFieldHtml = '<div class="mb-4 row"><label class="col-form-label col-lg-2">' + argumentName['name'] + '</label>' +
                '<div class="col-lg-10"><input class="form-control" type="text" name="optimizer_argument_' + argumentName['name'] + '" value="' + defaultValue + '"></div></div>';
            $('#optimizer-arguments').append(inputFieldHtml);  // Append input fields
            $('#optimizer-arguments').addClass('p-4 mb-3');
        }
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
        if ('name' in argumentName) {
            var defaultValue = argumentName['default'];
            var inputFieldHtml = '<div class="mb-4 row"><label class="col-form-label col-lg-2">' + argumentName['name'] + '</label>' +
                '<div class="col-lg-10"><input class="form-control" type="text" name="loss_argument_' + argumentName['name'] + '" value="' + defaultValue + '"></div></div>';
            $('#loss-arguments').append(inputFieldHtml);  // Append input fields
            $('#loss-arguments').addClass('p-4 mb-3');
        }
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
        if ('name' in argumentName) {
            var defaultValue = argumentName['default'];
            var inputFieldHtml = '<div class="mb-4 row"><label class="col-form-label col-lg-2">' + argumentName['name'] + '</label>' +
                '<div class="col-lg-10"><input class="form-control" type="text" name="layer_argument_' + argumentName['name'] + '" value="' + defaultValue + '"></div></div>';
            $('#layer-arguments').append(inputFieldHtml);  // Append input fields
            $('#layer-arguments').addClass('p-4 mb-3');
        }
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
    }

    if (selectedMetrics.length === 0) {
        return ;
    }

    for (const selectedMetric of selectedMetrics) {
        if (selectedMetric.jsonConfig.length > 0) {
                
            let metricContainerHTML = '<div class="card shadow-none m-3 w-auto" style="min-width:33%; max-width: 50%"> <div class="card-header">' + selectedMetric.name + '</div><div class="card-body">';
            
            for (const argumentName of selectedMetric.jsonConfig) {
                if ('name' in argumentName) {
                    var defaultValue = argumentName['default'];
                    var inputFieldHtml = '<div class="mb-4 row"><label class="col-form-label col-lg-4">' + argumentName['name'] + '</label>' +
                    '<div class="col-lg-8"><input class="form-control" type="text" name="metric_argument_' + selectedMetric.id + '_' + argumentName['name'] + '" value="' + defaultValue + '"></div></div>';
                    metricContainerHTML += inputFieldHtml;
                }
            }
            
            metricContainerHTML += '</div></div>';
            $('#metrics-arguments').append(metricContainerHTML);  // Append input fields
            $('#metrics-arguments').addClass('p-4 mb-3');
            
            if (metric_weight) {
                weight_html = '<div class="mb-4 row"><label class="col-form-label col-lg-4">' + selectedMetric.name + '</label>' +
                '<div class="col-lg-8"><input class="form-control" type="text" name="metric_weight_' + selectedMetric.id + '" value="1.0"></div></div>';
                metric_weight.append(weight_html);
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
        if ('name' in argumentName) {
            var defaultValue = argumentName['default'];
            var inputFieldHtml = '<div class="mb-4 row"><label class="col-form-label col-lg-2">' + argumentName['name'] + '</label>' +
                '<div class="col-lg-10"><input class="form-control" type="text" name="tuner_argument_' + argumentName['name'] + '" value="' + defaultValue + '"></div></div>';
            $('#tuner-arguments').append(inputFieldHtml);  // Append input fields
            $('#tuner-arguments').addClass('p-4 mb-3');
        }
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
        if ('name' in argumentName) {
            var defaultValue = argumentName['default'];
            var inputFieldHtml = '<div class="mb-4 row"><label class="col-form-label col-lg-2">' + argumentName['name'] + '</label>' +
                '<div class="col-lg-10"><input class="form-control" type="text" name="layer_argument_' + argumentName['name'] + '" value="' + defaultValue + '"></div></div>';
            $('#layer-arguments').append(inputFieldHtml);  // Append input fields
            $('#layer-arguments').addClass('p-4 mb-3');
        }
    }
}
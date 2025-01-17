function addRunToComparisonView(id, name, runType, link) {
    var container = $('#comparison_container');
    var newContainer = $('<div>', {
        class: 'border-bottom p-3 d-flex align-items-center justify-content-between'
    });
    newContainer.id = 'comparison_' + id + '_' + runType;
    var linkElement = $("<a>", {
        text: name,
        href: link,
        class: 'pr-3'
    });
    var deleteButton = $('<button>', {
        class: 'bg-danger button',
        click: removeComparison
    });
    deleteButton.attr('data-id', id);
    var icon = $('<i>', {
        class: 'fa fa-trash'
    });
    deleteButton.append(icon);
    newContainer.append(linkElement);
    newContainer.append(deleteButton);
    container.append(newContainer);

}

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

function handleTensorflowModelChange(selectElement) {
    var selectedModelId = $(selectElement).val();
    var selectedModel = modelOptions.find(function(opt) {
        return opt.id == selectedModelId;
    });

    $('#tensorflow_model-arguments').empty();
    $('#tensorflow_model-arguments').removeClass('p-4 mb-3');
    
    if (!selectedModel) {
        return ;
    }
    // Display input fields for each argument
    for (const argumentName of selectedModel.jsonConfig) {
        $('#tensorflow_model-arguments').append(addInputField(argumentName, 'tensorflow_model_argument_'));  // Append input fields
        $('#tensorflow_model-arguments').addClass('p-4 mb-3');
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
        inputName = prefix + argument['name'];
        var dtype = argument['dtype'];
        var inputField = '<div class="mb-4 row col-12"><label class="col-form-label col-lg-3">' + label + '</label>' +
        '<div class="col-lg-9 d-flex">';
        switch (dtype) {
            case "int":
                inputField += '<input class="form-control" type="number" step="1" id="' + inputName+ '" name="' + inputName + '" value="' + defaultValue + '">';
                break;
            case "float":
                inputField += '<input class="form-control" type="number" step="any" id="' + inputName+ '" name="' + inputName + '" value="' + defaultValue + '">';
                break;
            case "bool":
                inputField += '<input type="hidden" name="' + inputName + '" value="False"><input class="form-control" type="checkbox" name="' + inputName + '"';
                if (defaultValue && defaultValue.toString().toLowerCase() == 'true') {
                    inputField += ' checked="checked"';
                }
                inputField +=' value="True">';
                break;
            default:
                if (dtype.startsWith("ENUM")) {
                    arguments = convertStringToList(dtype.substr(4));
                    inputField += '<select class="form-control" id="' + inputName+ '" name="' + inputName + '">';
                    for (const arg of arguments) {
                        inputField+= "<option";
                        if (defaultValue === arg) {
                            inputField += " selected";
                        }
                        inputField += ">" + arg + "</option>";
                    }
                    inputField += "</select>"
                    break;
                } else if (dtype.startsWith("tuple")) {
                    inputField += '<input data-type="' + argument['dtype'] + '" class="form-control" type="text" id="' + inputName+ '" name="' + inputName + '" value="' + defaultValue + '" onInput="validateTupleInput(this, \'' + argument['dtype'] + '\')">';
                    break;
                } else {
                    inputField += '<input class="form-control" type="text" id="' + inputName+ '" name="' + inputName + '" value="' + defaultValue + '">';
                    break;
                }
        }
        if ((dtype !== 'unknown' && dtype !== 'NoneType' && !dtype.startsWith("ENUM")) || 'help' in argument) {
            
            inputField += '<span class="icon"><i class="fa fa-help icon" data-tooltip="Type: ' + argument['dtype'];
            if ('help' in argument) {
                inputField += '\n ' + argument['help'];
            }
            inputField += '">?</i></span>';
        } else {
            inputField += '<span class="icon"></span>';
        }
        inputField += '</div></div>';
        return inputField
    }
    return '';   
}

function containsDictWithValueForKey(listOfDicts, key, value, removeDuplicate) {
    // Iterate over each dictionary in the list
    for (let i = 0; i < listOfDicts.length; i++) {
        let dict = listOfDicts[i];
        // Check if the dictionary contains the key and has the specified value
        if (dict[key] === value) {
            if (removeDuplicate) {
                listOfDicts.splice(i, 1); // Remove the dictionary from the list
            }
            return true; // Dictionary with the specified key-value pair found
        }
    }
    return false; // No dictionary with the specified key-value pair found
}

function setArgumentValue(argumentType, name, value) {
    var element = document.getElementsByName(argumentType + name)[0];
    if (!element) {
        return;
    }
    if (element.type === 'hidden') {
        // This is a checkbox so we need to set values differently
        var checkbox = document.getElementsByName(argumentType + name)[1];
        if (value && value.toString().toLowerCase() == "true") {
            checkbox.checked = true;
        }
    } else {
        element.value = value;        
    }
}

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function snake_case_string(str) { 
    return str && str.match( 
/[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+/g) 
        .map(s => s.toLowerCase()) 
        .join('_'); 
}
function fitAxesToData(chart) {
    // Get the datasets from the chart
    const datasets = chart.data.datasets;

    // Initialize min and max values for x and y axes
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    // Loop through each dataset to find min and max values
    datasets.forEach(dataset => {
        dataset.data.forEach(point => {
            // Update min and max values for x axis
            if (point.x < Number(minX)) minX = Number(point.x);
            if (point.x > Number(maxX)) maxX = Number(point.x);

            // Update min and max values for y axis
            if (point.y < Number(minY)) minY = Number(point.y);
            if (point.y > Number(maxY)) maxY = Number(point.y);
        });
    });
    let paddingFactor = 0.02;
    const xPadding = (maxX - minX) * paddingFactor;
    const yPadding = (maxY - minY) * paddingFactor;

    // Apply padding
    minX -= xPadding;
    maxX += xPadding;
    minY -= yPadding;
    maxY += yPadding;
    

    // Update the chart's scales with new min and max values
    chart.options.scales.x.min = minX;
    chart.options.scales.x.max = maxX;
    chart.options.scales.y.min = minY;
    chart.options.scales.y.max = maxY;

    // Update the chart
    chart.update();
}

function calculateRegressionLine(data) {
                var sumX = 0;
    var sumY = 0;
    var sumXY = 0;
    var sumX2 = 0;

    var n = data.length;

    for (var i = 0; i < n; i++) {
        sumX += data[i].x;
        sumY += data[i].y;
        sumXY += data[i].x * data[i].y;
        sumX2 += data[i].x ** 2;
    }

    var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX ** 2);
    var intercept = (sumY - slope * sumX) / n;

    return { slope: slope, intercept: intercept };

}

function convertStringToList(str) {
    // Remove the parentheses
    str = str.replace(/[()]/g, '');

    // Split the string by ', ' and trim any extra spaces
    const list = str.split(', ').map(item => item.trim());

    return list;
}

function isValidTupleDetailed(str,requiredParts) {
    if (str === 'undefined') {
        return true;
    }
    if (!str.startsWith('(') || !str.endsWith(')')) {
        return false;
    }

    const content = str.slice(1, -1).trim();  // Remove the parentheses
    const parts = content.split(',').map(part => part.trim());  // Split and trim spaces

    if (parts.length !== requiredParts.length) {
        return false;
    }

    for (const partIdx in parts) {
        part = parts[partIdx];
        partType = requiredParts[partIdx];
        if (part.length === 0) {
            console.log(part + " is short");
            return false;
        }
        switch(partType) {
            case "int":
                if (!isInteger(part)) {
                    console.log(part + " is not an int");
                    return false;
                }
                break;
            case "float":
                if (!isFinite(part)) {
                    return false;
                }
                break;
            case "bool":
                if (part.toLowerCase() !== 'false' && part.toLowerCase() !== 'true') {
                    return false;
                }
                break;
            default:
                break;
        }
    }

    return true;
}

function isInteger(value) {
    return /^\d+$/.test(value);
}

function validateTupleInput(elm, dtype) {
    var parts = convertStringToList(dtype.substr(5));
    var isValid = isValidTupleDetailed(elm.value, parts);
    if (!isValid) {
        elm.classList.add('border-danger');
    } else {
        elm.classList.remove('border-danger');
    }
}

metricWeights = new Map();

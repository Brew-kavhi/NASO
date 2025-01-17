
sigma.utils.pkg('sigma.canvas.nodes');
sigma.canvas.nodes.image = (function () {
    // Return the renderer itself:
    var renderer = function (node, context, settings) {
        const nodeX = node['renderer1:x'];
        const nodeY = node['renderer1:y'];
        const label = node.id;
        const labelPadding = 10; // Adjust the padding as needed

        // Calculate the width based on the label length and padding
        const boxWidth = context.measureText(label).width + 2 * labelPadding;

        // const boxWidth = 60;
        const boxHeight = 34;
        context.fillStyle = settings('fadedNodeColor');

        context.strokeStyle = settings('defaultNodeColor'); // Border color
        context.lineWidth = 2; // Adjust border width as needed
        // Fill the rectangle with the white background
        context.fillRect(nodeX, nodeY - boxHeight / 2, boxWidth, boxHeight);
        // context.fillRect(nodeX - boxWidth / 2, nodeY - boxHeight / 2, boxWidth, boxHeight);

        // Draw the border
        context.strokeRect(nodeX, nodeY  - boxHeight / 2, boxWidth, boxHeight);
        // context.strokeRect(nodeX - boxWidth / 2, nodeY - boxHeight / 2, boxWidth, boxHeight);

        context.fillStyle = '#000';
        context.font = '15px Arial';
        node.label ='';
        context.fillText(node.id, nodeX + 5, nodeY  + 5);
    };
    return renderer;
})();


const s = new sigma({
    renderer: {
        container: document.getElementById('graph-container'),
        type: sigma.renderers.canvas
    },
    container: document.getElementById('graph-container'),
    settings: {
        minEdgeSize: 0.1,
        maxEdgeSize: 2,
        minNodeSize: 1,
        // edgeLabelSize: 'proportional',
        enableHovering: false,
        maxNodeSize: 8,
        defaultNodeColor: '#0069d9',
        fadedNodeColor:'#d3e8ff',
        minArrowSize: 10
    }
}
);



// Data structure to store nodes and edges
const nodes = new Map();
nodes.set('input_node', { id: "input_node", label: "Input", x: 0, y: 0, size: 3, color: '#008cc2', type: 'image' });
// const display_nodes = nodes;
let edges = [];




// gets the arguments fpr the layer and returns a dict
function getLayerArguments() {
    const form = document.getElementById('start_new_run');
    const inputs = form.elements;

    const values = [];

    for (let i = 0; i < inputs.length; i++) {
        const input = inputs[i];

        if (input.name && input.name.startsWith('layer_argument_')) {
            const key = input.name.replace('layer_argument_', '');
            const value = input.value;
            if (input.type == 'checkbox' && input.checked) {
                containsDictWithValueForKey(values, 'name', key, true);
                values.push({'name': key, 'value': value});
            }
            else if (!containsDictWithValueForKey(values, 'name', key, false)) {
                values.push({ 'name': key, 'value': value });
            }
        }
    }

    return values;
}

// Function to add a new node to the graph
function addNode() {
    // TODO when setting the nodes, all the information is destroyed
    existingNodes = s.graph.nodes();
    nodes.clear();
    for (const node of existingNodes) {
        nodes.set(node.id, node);
    }
    const selectedNodes = Array.from(document.getElementById('existing_nodes').selectedOptions).map(option => option.value);
    const layerType ={value: $('#id_layers').val(), text: layerOptions.find((layer)=>layer.id===Number($('#id_layers').val())).name};
    const nodeName = `${layerType.text} (${nodes.size})`;
    const nodeId = `${layerType.text}_${nodes.size}`;
    var yValue = nodes.size;
    var xValue = 0;
    
    if (selectedNodes) {
        var parentNode = nodes.get(selectedNodes[0]);
        if (parentNode) {
            yValue = parentNode.y + 1;
            xValue = parentNode.x;
        }
    }

    if (nodeName && !nodes.has(nodeName)) {
        // Create a new node
        const newNode = {
            id: nodeId,
            label: nodeName,
            x: xValue,
            y: yValue,
            size: 3,
            color: '#008cc2',
            naso_type: layerType.value,
            type:'image',
            additional_arguments: getLayerArguments(),
        };

        // Add edges to selected nodes
        buildEdgesForNode(nodeId, selectedNodes);

        // Add the new node to the graph
        nodes.set(nodeId, newNode);
        nodes.set(nodeId, newNode);

        // Refresh the graph
        refreshGraph(nodeId);
    }
}

function updateNode() {
    container = document.getElementById('networkgraph');
    addButton = container.querySelector('[name="addnode"]');
    updateButton = container.querySelector('[name="updatenode"]');
    deleteButton = container.querySelector('[name="deletenode"]');

    addButton.classList.remove('d-none');
    updateButton.classList.add('d-none');
    deleteButton.classList.add('d-none');

    header = container.querySelector('#node_header');
    header.innerHTML = 'Details';

    const nodeId = deleteButton.getAttribute('data-node-id');
    let updatingNode = nodes.get(nodeId);

    // set new properties
    const layerType ={value: $('#id_layers').val(), text: layerOptions.find((layer)=>layer.id===Number($('#id_layers').val())).name};
    const nodeName = `${layerType.text}_` + nodeId.split('_')[1];
    const selectedNodes = Array.from(document.getElementById('existing_nodes').selectedOptions).map(option => option.value);

    updatingNode.naso_type = layerType.value;
    updatingNode.additional_arguments = getLayerArguments();

    //delete all edges:
    deleteEdgesForNode(nodeId, nodeName);

    // rebiuld the edges:
    buildEdgesForNode(nodeName, selectedNodes);
    if (nodeId !== nodeName) {
        nodes.delete(nodeId);
        nodes.set(nodeName, {...updatingNode, id: nodeName});
    }
    refreshGraph();
}

function buildEdgesForNode(nodeId, parentNodes) {
    parentNodes.forEach(parentNode => {
        edges.push({
            id: `${nodeId}-${parentNode}`,
            source: parentNode,
            target: nodeId,
            type:'arrow'
        });
    });
}

function deleteEdgesForNode(nodeId, newNodeName) {
    const edgesToSplice = [];
    for (let i = 0; i < edges.length; i++) {
        if (edges[i].target === nodeId) {
            edgesToSplice.push(i);
        } else if(edges[i].source === nodeId) {
            edges[i].source = newNodeName;
        }
    }

    // Remove the edges using splice
    for (let i = edgesToSplice.length - 1; i >= 0; i--) {
        const index = edgesToSplice[i];
        edges.splice(index, 1);
    }
}

function deleteNode() {
    if (confirm("Diese Ebene wirklich loschen?")) {
        container = document.getElementById('networkgraph');
        addButton = container.querySelector('[name="addnode"]');
        updateButton = container.querySelector('[name="updatenode"]');
        deleteButton = container.querySelector('[name="deletenode"]');

        addButton.classList.remove('d-none');
        updateButton.classList.add('d-none');
        deleteButton.classList.add('d-none');

        header = container.querySelector('#node_header');
        header.innerHTML = 'Details';

        const nodeId = deleteButton.getAttribute('data-node-id');

        // delete the node and the edges
        nodes.delete(nodeId);
        deleteEdgesForNode(nodeId);
        refreshGraph();
    }
}

// Function to refresh the graph with updated nodes and edges
function refreshGraph(nodeId = undefined) {
    s.graph.clear();
    s.graph.read({ nodes: [...nodes.values()], edges });
    s.refresh();

    const selectBox = document.getElementById('existing_nodes');

    if(selectBox) {
        // Clear the current options in the select box
        selectBox.innerHTML = '';
        // Iterate through all nodes and add them as options
        nodes.forEach((node, nodeName) => {
            const option = document.createElement('option');
            option.value = node.id;
            option.text = node.id;
            selectBox.appendChild(option);
        });
        if (nodeId) {
            selectBox.value = nodeId;
        }
        document.getElementById("architecture_nodes").value = JSON.stringify([...nodes.values()]);
        document.getElementById("architecture_edges").value = JSON.stringify(edges);
    }
    s.bind('clickNode', function (event) {
        const clickedNodeId = event.data.node.id;
        const clickedNode = nodes.get(clickedNodeId);

            // get the buttons and hide the addNodebutton
        container = document.getElementById('networkgraph');
        if (container) {
            addButton = container.querySelector('[name="addnode"]');
            updateButton = container.querySelector('[name="updatenode"]');
            deleteButton = container.querySelector('[name="deletenode"]');
        
            addButton.classList.add('d-none');
            updateButton.classList.remove('d-none');
            deleteButton.classList.remove('d-none');

            updateButton.setAttribute('data-node-id', clickedNodeId);
            deleteButton.setAttribute('data-node-id', clickedNodeId);

            header = container.querySelector('#node_header');
            header.innerHTML = clickedNode.id;
        }
            // set value of the layer type
            $('#id_layers').val(clickedNode.naso_type);
            $('#id_layers').trigger('change');

            // set value of selected nodes:
            parentNodes = []
            for (const edge of edges) {
                if (edge.target === clickedNode.id) {
                    parentNodes.push(edge.source);
                }
            }
            $('#existing_nodes').val(parentNodes);
            $('#existing_nodes').trigger('change');

            // set additional arguments of the layer
            for (argument of clickedNode.additional_arguments) {
                setArgumentValue('layer_argument_', argument.name, argument.value);
            }
        });

    sigma.plugins.dragNodes(s, s.renderers[0]);
    // Event handler for node click
}

// Initial graph rendering
refreshGraph();

var
	// Sequential
	Spectral3 = ['#fc8d59', '#ffffbf', '#99d594'],
	Spectral4 = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba'],
	Spectral5 = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba'],
	Spectral6 = ['#d53e4f', '#fc8d59', '#fee08b', '#e6f598', '#99d594', '#3288bd'],
	Spectral7 = ['#d53e4f', '#fc8d59', '#fee08b', '#ffffbf', '#e6f598', '#99d594', '#3288bd'],
	Spectral8 = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'],
	Spectral9 = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'],
	Spectral10 = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'],
	Spectral11 = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'],

	RdYlGn3 = ['#fc8d59', '#ffffbf', '#91cf60'],
	RdYlGn4 = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'],
	RdYlGn5 = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
	RdYlGn6 = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'],
	RdYlGn7 = ['#d73027', '#fc8d59', '#fee08b', '#ffffbf', '#d9ef8b', '#91cf60', '#1a9850'],
	RdYlGn8 = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850'],
	RdYlGn9 = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850'],
	RdYlGn10 = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'],
	RdYlGn11 = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'],

	// Qualitative
	Accent3 = ['#7fc97f', '#beaed4', '#fdc086'],
	Accent4 = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99'],
	Accent5 = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0'],
	Accent6 = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f'],
	Accent7 = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17'],
	Accent8 = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17', '#666666'],

	DarkTwo3 = ['#1b9e77', '#d95f02', '#7570b3'],
	DarkTwo4 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a'],
	DarkTwo5 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e'],
	DarkTwo6 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02'],
	DarkTwo7 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d'],
	DarkTwo8 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'],

	Paired3 = ['#a6cee3', '#1f78b4', '#b2df8a'],
	Paired4 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'],
	Paired5 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99'],
	Paired6 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c'],
	Paired7 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f'],
	Paired8 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00'],
	Paired9 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6'],
	Paired10 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a'],
	Paired11 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99'],
	Paired12 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'],

	PastelOne3 = ['#fbb4ae', '#b3cde3', '#ccebc5'],
	PastelOne4 = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4'],
	PastelOne5 = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6'],
	PastelOne6 = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc'],
	PastelOne7 = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd'],
	PastelOne8 = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec'],
	PastelOne9 = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2'],

	PastelTwo3 = ['#b3e2cd', '#fdcdac', '#cbd5e8'],
	PastelTwo4 = ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4'],
	PastelTwo5 = ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9'],
	PastelTwo6 = ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae'],
	PastelTwo7 = ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc'],
	PastelTwo8 = ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc'],

	SetOne3 = ['#e41a1c', '#377eb8', '#4daf4a'],
	SetOne4 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'],
	SetOne5 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'],
	SetOne6 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33'],
	SetOne7 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628'],
	SetOne8 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
	SetOne9 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'],

	SetTwo3 = ['#66c2a5', '#fc8d62', '#8da0cb'],
	SetTwo4 = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'],
	SetTwo5 = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'],
	SetTwo6 = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'],
	SetTwo7 = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494'],
	SetTwo8 = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'],

	SetThree3 = ['#8dd3c7', '#ffffb3', '#bebada'],
	SetThree4 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072'],
	SetThree5 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3'],
	SetThree6 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462'],
	SetThree7 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69'],
	SetThree8 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5'],
	SetThree9 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9'],
	SetThree10 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'],
	SetThree11 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5'],
	SetThree12 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'];

// ==========================================
// COASTAL DISASTER RISK PROJECT
// Study Area: Puri, Odisha
// ==========================================


// ------------------------------------------
// 1. DEFINE STUDY AREA
// ------------------------------------------

var studyArea = ee.Geometry.Rectangle([
  85.4,
  19.5,
  86.0,
  20.2
]);

Map.centerObject(studyArea, 10);


// ------------------------------------------
// 2. ELEVATION
// ------------------------------------------

var elevation = ee.Image(
  'USGS/SRTMGL1_003'
);

Map.addLayer(
  elevation.clip(studyArea),
  {
    min: 0,
    max: 100,
    palette: [
      'blue',
      'cyan',
      'yellow',
      'red'
    ]
  },
  'Elevation'
);


// ------------------------------------------
// 3. LAND COVER
// ------------------------------------------

var landCover = ee.ImageCollection(
  'ESA/WorldCover/v200'
).first();

var lc = landCover.select('Map');

Map.addLayer(
  lc.clip(studyArea),
  {
    min: 10,
    max: 100,
    palette: [
      '006400',
      'ffbb22',
      'ffff4c',
      'f096ff',
      'fa0000',
      'b4b4b4',
      'f0f0f0',
      '0064c8',
      '0096a0',
      '00cf75',
      'fae6a0'
    ]
  },
  'Land Cover'
);


// ------------------------------------------
// 4. RAINFALL
// ------------------------------------------

var rainfall = ee.ImageCollection(
  'ECMWF/ERA5_LAND/HOURLY'
)
.filterDate(
  '2024-10-01',
  '2024-10-03'
)
.select(
  'total_precipitation'
);

var totalRainfall = rainfall
  .sum()
  .multiply(1000);

Map.addLayer(
  totalRainfall.clip(studyArea),
  {
    min: 0,
    max: 200,
    palette: [
      'white',
      'cyan',
      'blue',
      'purple'
    ]
  },
  'Rainfall (mm)'
);


// ------------------------------------------
// 5. WIND
// ------------------------------------------

var weather = ee.ImageCollection(
  'ECMWF/ERA5_LAND/HOURLY'
)
.filterDate(
  '2024-10-01',
  '2024-10-03'
)
.select([
  'u_component_of_wind_10m',
  'v_component_of_wind_10m'
]);

var windSpeed = weather.map(function(image) {

  var u = image.select(
    'u_component_of_wind_10m'
  );

  var v = image.select(
    'v_component_of_wind_10m'
  );

  var speed = u
    .pow(2)
    .add(v.pow(2))
    .sqrt()
    .rename('wind_speed');

  return speed;
});

var maxWind = windSpeed.max();

Map.addLayer(
  maxWind.clip(studyArea),
  {
    min: 0,
    max: 40,
    palette: [
      'white',
      'cyan',
      'blue',
      'yellow',
      'orange',
      'red'
    ]
  },
  'Maximum Wind Speed'
);


// ------------------------------------------
// 6. HISTORICAL WATER
// ------------------------------------------

var water = ee.Image(
  'JRC/GSW1_4/GlobalSurfaceWater'
);

var occurrence = water.select(
  'occurrence'
);


// Water occurring more than 50% of the time
var waterMask = occurrence.gt(50);


// Display water
Map.addLayer(
  waterMask.clip(studyArea),
  {
    min: 0,
    max: 1,
    palette: [
      'white',
      'blue'
    ]
  },
  'Historical Water'
);


// ------------------------------------------
// 7. STUDY AREA BORDER
// ------------------------------------------

Map.addLayer(
  studyArea,
  {
    color: 'red'
  },
  'Study Area'
);// ===============================
// HISTORICAL FLOOD LABEL
// ===============================

var floods = ee.ImageCollection('GLOBAL_FLOOD_DB/MODIS_EVENTS/V1')
  .filterBounds(studyArea);

print('Number of flood events:', floods.size());

// Combine all historical flood events
var floodMask = floods
  .select('flooded')
  .max()
  .gt(0);

Map.addLayer(
  floodMask.clip(studyArea),
  {
    min: 0,
    max: 1,
    palette: ['white', 'red']
  },
  'Historical Flood Areas'
);
// ===============================
// ML FEATURE STACK
// ===============================

// Elevation
var elevationFeature = elevation
  .rename('elevation');

// Rainfall
var rainfallFeature = totalRainfall
  .rename('rainfall');

// Wind
var windFeature = maxWind
  .rename('wind');

// Historical water
var waterFeature = waterMask
  .rename('historical_water');

// Land cover
var landCoverFeature = lc
  .rename('land_cover');

// Combine all features
var featureStack = ee.Image.cat([
  elevationFeature,
  rainfallFeature,
  windFeature,
  waterFeature,
  landCoverFeature
]);

print('Feature stack:', featureStack);
// ===============================
// CREATE ML TRAINING DATASET
// ===============================

// Add the flood label to our features
var trainingImage = featureStack.addBands(
  floodMask.rename('flooded')
);

// Sample pixels from the study area
var trainingData = trainingImage.sample({
  region: studyArea,
  scale: 1000,
  numPixels: 5000,
  seed: 42,
  geometries: true
});

print('Training dataset:', trainingData);
print('Number of training samples:', trainingData.size());
// ===============================
// CHECK FLOOD LABEL DISTRIBUTION
// ===============================

var floodedSamples = trainingData.filter(
  ee.Filter.eq('flooded', 1)
);

var nonFloodedSamples = trainingData.filter(
  ee.Filter.eq('flooded', 0)
);

print(
  'Flooded samples:',
  floodedSamples.size()
);

print(
  'Non-flooded samples:',
  nonFloodedSamples.size()
);
// ===============================
// TRAIN XGBOOST FLOOD MODEL
// ===============================

var classifier = ee.Classifier.smileGradientTreeBoost({
  numberOfTrees: 100,
  shrinkage: 0.1,
  samplingRate: 0.7,
  maxNodes: 20,
  loss: 'LogLoss',
  seed: 42
}).train({
  features: trainingData,
  classProperty: 'flooded',
  inputProperties: [
    'elevation',
    'rainfall',
    'wind',
    'historical_water',
    'land_cover'
  ]
});

print('XGBoost model:', classifier);
// ===============================
// PREDICT FLOOD RISK
// ===============================

var riskMap = featureStack.classify(classifier);

print('Risk map:', riskMap);

Map.addLayer(
  riskMap.clip(studyArea),
  {
    min: 0,
    max: 1,
    palette: ['green', 'yellow', 'red']
  },
  'Predicted Flood Risk'
);
// ===============================
// TRAIN / TEST SPLIT
// ===============================

var randomData = trainingData.randomColumn('random', 42);

var trainData = randomData.filter(
  ee.Filter.lt('random', 0.7)
);

var testData = randomData.filter(
  ee.Filter.gte('random', 0.7)
);

print('Training samples:', trainData.size());
print('Testing samples:', testData.size());
// ===============================
// ===============================
// XGBOOST PROBABILITY MODEL
// ===============================

var probabilityClassifier =
  ee.Classifier.smileGradientTreeBoost({
    numberOfTrees: 100,
    shrinkage: 0.1,
    samplingRate: 0.7,
    maxNodes: 20,
    loss: 'LogLoss',
    seed: 42
  })
  .setOutputMode('PROBABILITY')
  .train({
    features: trainData,
    classProperty: 'flooded',
    inputProperties: [
      'elevation',
      'rainfall',
      'wind',
      'historical_water',
      'land_cover'
    ]
  });

print(
  'Probability XGBoost model:',
  probabilityClassifier
);
// ===============================
// FLOOD PROBABILITY MAP
// ===============================

var probabilityMap =
  featureStack.classify(probabilityClassifier);

Map.addLayer(
  probabilityMap.clip(studyArea),
  {
    min: 0,
    max: 1,
    palette: [
      'green',
      'yellow',
      'orange',
      'red'
    ]
  },
  'Flood Probability'
);
// ===============================
// ===============================
// HEALTHCARE ACCESSIBILITY
// ===============================

var healthcareAccess = ee.Image(
  'projects/malariaatlasproject/assets/accessibility/accessibility_to_healthcare/2019'
);

var healthcareTime = healthcareAccess
  .select('accessibility');

Map.addLayer(
  healthcareTime.clip(studyArea),
  {
    min: 0,
    max: 120,
    palette: ['green', 'yellow', 'orange', 'red']
  },
  'Healthcare Accessibility'
);
// ===============================
// POWER INFRASTRUCTURE
// ===============================

var powerPlants = ee.FeatureCollection(
  'WRI/GPPD/power_plants'
);

// Keep only power plants inside our study area
var localPowerPlants = powerPlants.filterBounds(studyArea);

print(
  'Power plants in study area:',
  localPowerPlants.size()
);

// Display power plants
Map.addLayer(
  localPowerPlants,
  {
    color: 'black'
  },
  'Power Infrastructure'
);
// ===============================
// BUILT-UP INFRASTRUCTURE
// ===============================

var builtUp = lc.eq(50);

Map.addLayer(
  builtUp.clip(studyArea),
  {
    min: 0,
    max: 1,
    palette: ['white', 'black']
  },
  'Built-up Areas'
);

print(
  'Built-up area:',
  builtUp.selfMask().reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: studyArea,
    scale: 100,
    maxPixels: 1e9
  })
);
// ===============================
// OVERALL COASTAL RISK
// ===============================

// 1. Flood probability
var floodRisk = probabilityMap;

// 2. Healthcare vulnerability
// Higher travel time = higher vulnerability
var healthcareRisk = healthcareTime
  .divide(120)
  .clamp(0, 1);

// 3. Built-up exposure
var builtUpRisk = builtUp;

// Combine the three components
var overallRisk = floodRisk
  .multiply(0.6)
  .add(
    healthcareRisk.multiply(0.25)
  )
  .add(
    builtUpRisk.multiply(0.15)
  )
  .rename('overall_risk');

// Display overall risk
Map.addLayer(
  overallRisk.clip(studyArea),
  {
    min: 0,
    max: 1,
    palette: [
      'green',
      'yellow',
      'orange',
      'red'
    ]
  },
  'OVERALL COASTAL RISK'
);

print(
  'Overall Risk Map:',
  overallRisk
);
// ===============================
// RISK STATISTICS
// ===============================

var riskStats = overallRisk.reduceRegion({
  reducer: ee.Reducer.mean()
    .combine({
      reducer2: ee.Reducer.max(),
      sharedInputs: true
    })
    .combine({
      reducer2: ee.Reducer.min(),
      sharedInputs: true
    }),
  geometry: studyArea,
  scale: 1000,
  maxPixels: 1e9
});

print('Risk Statistics:', riskStats);
// ===============================
// RISK CATEGORIES
// ===============================

var riskCategory = overallRisk
  .expression(
    "(r < 0.30) ? 1" +
    " : (r < 0.60) ? 2" +
    " : (r < 0.80) ? 3" +
    " : 4",
    {
      r: overallRisk
    }
  )
  .rename('risk_category');

Map.addLayer(
  riskCategory.clip(studyArea),
  {
    min: 1,
    max: 4,
    palette: [
      'green',
      'yellow',
      'orange',
      'red'
    ]
  },
  'Risk Categories'
);
var componentStats = ee.Dictionary({
  flood_risk: probabilityMap.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea,
    scale: 1000,
    maxPixels: 1e9
  }).get('classification'),

  healthcare_risk: healthcareRisk.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea,
    scale: 1000,
    maxPixels: 1e9
  }).get('accessibility'),

  built_up_exposure: builtUpRisk.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea,
    scale: 1000,
    maxPixels: 1e9
  }).get('Map')
});

print('COMPONENT RISK:', componentStats);
var riskOutput = ee.Dictionary({
  location: 'Puri, Odisha',

  mean_risk: riskStats.get('overall_risk_mean'),
  max_risk: riskStats.get('overall_risk_max'),
  min_risk: riskStats.get('overall_risk_min'),

  flood_risk: componentStats.get('flood_risk'),
  healthcare_risk: componentStats.get('healthcare_risk'),
  built_up_exposure: componentStats.get('built_up_exposure')
});

print('FINAL GEMINI INPUT:', riskOutput);
// ===============================

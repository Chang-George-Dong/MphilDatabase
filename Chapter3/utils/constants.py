# List of all UEA/UCR archive dataset names
#UNIVARIATE_DATASET_NAMES_2018
UNIVARIATE_DATASET_NAMES = [
    'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'ArrowHead', 
    'BME', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'Chinatown', 'ChlorineConcentration', 
    'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 
    'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 
    'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'ECG200', 'ECG5000', 
    'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes', 'ElectricDevices', 
    'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 
    'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 
    'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan', 
    'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 
    'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound', 
    'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 
    'MedicalImages', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 
    'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain', 
    'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OSULeaf', 'OliveOil', 'PLAID', 
    'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure',
    'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 
    'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2', 
    'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll', 
    'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 
    'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 
    'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 
    'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine', 
    'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga',
]

# Name of the classifier used
CLASSIFIERS = ['inception']
# List of all univariate archives
UNIVARIATE_ARCHIVE_NAMES = ['UCRArchive_2018']
#UNIVARIATE_DATASET_NAMES_2018 = ['UCRArchive_2018']
#dataset_names_for_archive = {'UCRArchive_2018': UNIVARIATE_DATASET_NAMES_2018}
dataset_names_for_archive = {'UCRArchive_2018': UNIVARIATE_DATASET_NAMES}

# List of all the attack names
ATTACK_NAMES = ["GM_e","GM_l1","GM_kl","GM_l2","GM","SGM","GM_ratio"]  #['bim', 'gm', 'gm-wo-clip', 'sgm-wo-clip']
#["KL_l2l1","KL_l2",'KL',"GM_l2","GM","BIM","FGSM","SGM"]
#["KL_l2l1","KL_l2",'KL',"GM_l2","GM","BIM","FGSM","SGM"]
ATTACK_NAMES = ["GM_ratio_T"]  
NB_ITERATIONS = 1
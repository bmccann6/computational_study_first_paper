item_values = {"cocaine": 100,
                "heroin": 300,
                "meth": 94}     # The keys are drugs (e.g. "cocaine") and the value of the key is the value of the drugs per unit of shipment. 

resource_sets = {
    "2012": {"cocaine": 37, "heroin": 12, "meth": 88},
    "2013": {"cocaine": 45, "heroin": 15, "meth": 92},
    "2014": {"cocaine": 50, "heroin": 18, "meth": 85},
    "2015": {"cocaine": 55, "heroin": 20, "meth": 80},
    "2016": {"cocaine": 60, "heroin": 25, "meth": 75},
    "2017": {"cocaine": 65, "heroin": 30, "meth": 70},
    "2018": {"cocaine": 70, "heroin": 35, "meth": 65},
    "2019": {"cocaine": 75, "heroin": 40, "meth": 60},
    "2020": {"cocaine": 80, "heroin": 45, "meth": 55},
    "2021": {"cocaine": 85, "heroin": 50, "meth": 50},
    "2022": {"cocaine": 90, "heroin": 55, "meth": 45},
    "2023": {"cocaine": 95, "heroin": 60, "meth": 40},
    "2024": {"cocaine": 100, "heroin": 65, "meth": 35}
}       # Outer-keys are years. The keys in the inner-dictionaries are drugs. The values of those inner-keys are the number of shipment units of that drug into the United states in the year of the outer-key
                
# r = len(resource_sets) 
sizes_hiding_locations = [10, 8, 5, 5, 3]    #! CHANGE THIS LATER. JUST A PLACEHOLDER FOR NOW. At each index i, the size of hiding location i is the value.
detector_accuracies = [0.9, 0.8, 0.3, 0, 0]

required_num_samples_per_bin = 100



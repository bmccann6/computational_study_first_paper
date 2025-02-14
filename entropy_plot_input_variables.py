# item_values = {"cocaine": 100,
#                 "heroin": 300,
#                 "meth": 24}     # The keys are drugs (e.g. "cocaine") and the value of the key is the value of the drugs per unit of shipment. 

# resource_sets = {
#     "2012": {"cocaine": 37, "heroin": 12, "meth": 88},
#     "2013": {"cocaine": 45, "heroin": 15, "meth": 92},
#     "2014": {"cocaine": 50, "heroin": 18, "meth": 85},
#     "2015": {"cocaine": 55, "heroin": 20, "meth": 80},
#     "2016": {"cocaine": 60, "heroin": 25, "meth": 75},
#     "2017": {"cocaine": 65, "heroin": 30, "meth": 70},
#     "2018": {"cocaine": 70, "heroin": 35, "meth": 65},
#     "2019": {"cocaine": 75, "heroin": 40, "meth": 60},
#     "2020": {"cocaine": 80, "heroin": 45, "meth": 55},
#     "2021": {"cocaine": 85, "heroin": 50, "meth": 50},
#     "2022": {"cocaine": 90, "heroin": 55, "meth": 45},
#     "2023": {"cocaine": 95, "heroin": 60, "meth": 40},
#     "2024": {"cocaine": 100, "heroin": 65, "meth": 35}
# }       # Outer-keys are years. The keys in the inner-dictionaries are drugs. The values of those inner-keys are the number of shipment units of that drug into the United states in the year of the outer-key


# The following is the item values for our real data
item_values = { "meth": 11000,
                "cannabis": 2000,
                "cocaine": 29000,
                "heroin": 50000,
                "fentanyl": 50000}     # The keys are drugs (e.g. "cocaine") and the value of the key is the value of the drugs per unit of shipment. 

# The following is the quantity values for our real data
resource_sets = {
                 '2017': {'meth': 58013, 'cannabis': 693000, 'cocaine': 236966, 'heroin': 7917, 'fentanyl': 2158},
                 '2018': {'meth': 83341, 'cannabis': 510000, 'cocaine': 192080, 'heroin': 8996, 'fentanyl': 2525},
                 '2019': {'meth': 118284, 'cannabis': 432000, 'cocaine': 187018, 'heroin': 8824, 'fentanyl': 4489},
                 '2020': {'meth': 139989, 'cannabis': 555000, 'cocaine': 205354, 'heroin': 7191, 'fentanyl': 8319},
                 '2021': {'meth': 118473, 'cannabis': 338272, 'cocaine': 205354, 'heroin': 5222, 'fentanyl': 12227},
                 '2022': {'meth': 95793, 'cannabis': 288694, 'cocaine': 205354, 'heroin': 4140, 'fentanyl': 16123}
                 }
      # Outer-keys are years. The keys in the inner-dictionaries are drugs. The values of those inner-keys are the number of shipment units of that drug into the United states in the year of the outer-key
                
                
r = len(resource_sets) 

# These are the hiding locations based on Figure 8 of the "Port Performance Freight Statistics: 2025 Annual Report". 
# The keys are the imports. Also, we only do mainland US ports (not Alaska or Hawaii)
hiding_locations = { "Port authority of New York and New Jersey, NY & NJ": 5352 * 1000,
                     "Philadelphia regional port authority, PA": 536 * 1000,
                     "Wilmington, DE": 188 * 1000,
                     "Baltimore, MD": 522 * 1000,
                     "South jersey port corporation, NJ": 70 * 1000,
                     "Port of virginia, VA": 1810 * 1000,
                     "Wilmington, NC": 121 * 1000,
                     "Port of savannah, GA": 3013 * 1000,
                     "Port of charleston, SC": 1492 * 1000,
                     "Mobile, AL": 293 * 1000,
                     "Port of gulfport, MS": 94 * 1000,
                     "Jacksonville, FL": 212 * 1000,
                     "Port everglades, FL": 407 * 1000,
                     "Port miami, FL": 578 * 1000,
                     "San juan, PR": 217 * 1000,
                     "Port of new orleans, LA": 127 * 1000,
                     "Port of houston authority of harris county, TX": 2035 * 1000,
                     "Port of long beach, CA": 4676 * 1000,
                     "Port of los angeles, CA": 5232 * 1000,
                     "Oxnard harbor district, CA": 136 * 1000,
                     "Port of oakland, CA": 1032 * 1000,
                     "Tacoma, WA": 678 * 1000,
                     "Port of seattle, WA": 706 * 1000    
                    }

# For each detector, the keys are tuples, where the first element is the number of each detector we have, and the second element is the detection accuracy of one of those detectors.
detectors = { "German sheperd": (3, 0.868),
              "English cocker spaniel": (2, 0.82),
              "Labrador retriver": (2, 0.788),
              "Terrier": (1, 0.67),
              "X-ray scanner": (1, 0.8),
              "Chemical detection device": (1, 0.95),
              "Human": (3, 0.3)            
            }

percent_cargo_containers_storing_drugs = 0.01
sizes_hiding_locations = [int(size * percent_cargo_containers_storing_drugs) for size in sorted(hiding_locations.values(), reverse=True)]        # At each index i, the size of hiding location i is the value.
detector_accuracies = sorted([accuracy for count, accuracy in detectors.values() for _ in range(count)], reverse=True)

# sizes_hiding_locations = [50, 40, 35, 20, 20, 15, 10, 5, 3, 2]       # At each index i, the size of hiding location i is the value.
# detector_accuracies = [0.8, 0.8, 0.7, 0.5, 0.5, 0.3, 0.3, 0, 0, 0]   # Detector accuracies, given in descending order 

required_num_samples_per_bin = 50


Comment out necessary things in order to do testing. Dont use real-world data for testing.
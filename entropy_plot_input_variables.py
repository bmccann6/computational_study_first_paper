# item_vals = {"cocaine": 100,
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


# The following is the item values for our real data, sorted by descending price
item_vals = {
                "fentanyl": 50000,
                "heroin": 50000,
                "cocaine": 29000,
                "meth": 11000,
                "cannabis": 2000
            }

# The following is the quantity values for our real data, with keys listed in the same (descending) price order
resource_sets = {
                    "2017": {"fentanyl": 2158, "heroin": 7917, "cocaine": 236966, "meth": 58013, "cannabis": 693000},
                    "2018": {"fentanyl": 2525, "heroin": 8996, "cocaine": 192080, "meth": 83341, "cannabis": 510000},
                    "2019": {"fentanyl": 4489, "heroin": 8824, "cocaine": 187018, "meth": 118284, "cannabis": 432000},
                    "2020": {"fentanyl": 8319, "heroin": 7191, "cocaine": 205354, "meth": 139989, "cannabis": 555000},
                    "2021": {"fentanyl": 12227, "heroin": 5222, "cocaine": 205354, "meth": 118473, "cannabis": 338272},
                    "2022": {"fentanyl": 16123, "heroin": 4140, "cocaine": 205354, "meth": 95793, "cannabis": 288694}
                }
                
num_resource_sets = len(resource_sets) 

# These are the hiding locations based on Figure 8 of the "Port Performance Freight Statistics: 2025 Annual Report". 
# The keys are the imports. Also, we only do mainland US ports (not Alaska or Hawaii)
# There are 23 locations in total. I ordered them by their capacities.
hiding_locations = { "Port authority of New York and New Jersey, NY & NJ": 5352 * 1000,
                     "Port of los angeles, CA": 5232 * 1000,
                     "Port of long beach, CA": 4676 * 1000,
                     "Port of savannah, GA": 3013 * 1000,
                     "Port of houston authority of harris county, TX": 2035 * 1000,
                     "Port of virginia, VA": 1810 * 1000,
                     "Port of charleston, SC": 1492 * 1000,
                     "Port of oakland, CA": 1032 * 1000,
                     "Tacoma, WA": 678 * 1000,
                     "Port of seattle, WA": 706 * 1000,
                     "Philadelphia regional port authority, PA": 536 * 1000,
                     "Port miami, FL": 578 * 1000,
                     "Port everglades, FL": 407 * 1000,
                     "Baltimore, MD": 522 * 1000,
                     "San juan, PR": 217 * 1000,
                     "Jacksonville, FL": 212 * 1000,
                     "Mobile, AL": 293 * 1000,
                     "Port of new orleans, LA": 127 * 1000,
                     "Wilmington, NC": 121 * 1000,
                     "Oxnard harbor district, CA": 136 * 1000,
                     "Wilmington, DE": 188 * 1000,
                     "South jersey port corporation, NJ": 70 * 1000,
                     "Port of gulfport, MS": 94 * 1000
                    }

# For each detector, the keys are tuples, where the first element is the number of each detector we have, and the second element is the detection accuracy of one of those detectors.
detectors = { "German sheperd": (3, 0.868),
              "English cocker spaniel": (2, 0.82),
              "Labrador retriver": (2, 0.788),
              "Terrier": (1, 0.67),
              "X-ray scanner": (1, 0.8),
              "Chemical detection device": (1, 0.95),
              "Human": (5, 0.3)            
            }


num_items_per_year = {year: sum(resources.values()) for year, resources in resource_sets.items()}
max_num_items_across_years = max(num_items_per_year.values())
total_capacity = sum(hiding_locations.values())
fraction_cargo_containers_storing_drugs = max_num_items_across_years/total_capacity   # We will multiply all the capacities in hiding_locations by this value in other modules. It is useful because we will never need to have the total capacity be greater than the max capacity needed across all years. Also, by having a fraction we will multiply all capacities by later in other modules, we make the changes to capacities more uniform across locations.
print(f"This is fraction_cargo_containers_storing_drugs: {fraction_cargo_containers_storing_drugs}")

sizes_hiding_locations = [int(size * fraction_cargo_containers_storing_drugs) for size in sorted(hiding_locations.values(), reverse=True)]        # At each index i, the size of hiding location i is the value.
total_real_detectors = sum(count for count, _ in detectors.values())
null_detectors_count = len(sizes_hiding_locations) - total_real_detectors
detector_accuracies = sorted([accuracy for count, accuracy in detectors.values() for _ in range(count)] + [0] * null_detectors_count, reverse=True)

# sizes_hiding_locations = [50, 40, 35, 20, 20, 15, 10, 5, 3, 2]       # At each index i, the size of hiding location i is the value.
# detector_accuracies = [0.8, 0.8, 0.7, 0.5, 0.5, 0.3, 0.3, 0, 0, 0]   # Detector accuracies, given in descending order 

NUM_SAMPLES_NEEDED_PER_BIN = 100
NUM_BINS = 20


# Comment out necessary things in order to do testing. Dont use real-world data for testing.
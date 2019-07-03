from itertools import permutations

class ReplacementListComparer():
    def __init__(self, old_list, new_list, approved_replacement_dict, possible_replacements=6):
        self.NUMBER_OF_REPLACEMENTS = possible_replacements
        self.old_list = old_list
        self.new_list = new_list
        self.json_translation_dict = approved_replacement_dict
        self.json_translation_dict_values = approved_replacement_dict.values()
        self.json_translation_dict_keys = approved_replacement_dict.keys()

    def search_for_common_replacements_match(self, old_item):
        for i in range(self.NUMBER_OF_REPLACEMENTS):
            multi_replace_item = old_item
            perm = permutations([v[0] for v in self.json_translation_dict_values], i) 
            # use obtained permutations to search for approved replacements
            for perm_tuple in list(perm): 
                for n in range(len(perm_tuple)):
                    original_key = self._unmap_translation(perm_tuple[n])
                    multi_replace_item = multi_replace_item.replace(original_key, perm_tuple[n])
                    if old_item.replace(original_key, perm_tuple[n]) in self.new_list:
                        # replacement match
                        #print("FOUND replacement EXACT MATCH: {}; From: {}; Replaced: {} for {}".format(old_item.replace(original_key, perm_tuple[n]), old_item, original_key, perm_tuple[n]))
                        return old_item.replace(original_key, perm_tuple[n])
                    elif multi_replace_item in self.new_list:
                        # rp
                        #print("FOUND multi_replace_item EXACT MATCH: {}; From: {}; Replaced: {} for {}".format(multi_replace_item, old_item, original_key, perm_tuple[n]))
                        return multi_replace_item
        return None

    def _unmap_translation(self, translated):
        for k, vals in self.json_translation_dict.items():
            if vals[0] == translated:
                return k
        return None

    def compare(self):
        translation = {}
        not_found = []
        # for list of old datacuts
        for old_item in self.old_list:
            if old_item in self.new_list:
                # YoY match
                print("FOUND EXACT MATCH: {}".format(old_item))
                #translation[old_item] = old_item

                continue
            elif any(substring for substring in self.json_translation_dict_keys if substring in old_item ):
                answer = self.search_for_common_replacements_match(old_item)
                if answer is not None:
                    print("Answer: '{}'; Old: '{}'".format(answer, old_item))
                    translation[old_item] = answer
                else:
                    print("Could not find: {}".format(old_item))
                    not_found.append(old_item)
            else:
                print("Could not find: {}".format(old_item))
                not_found.append(old_item)
        return translation, not_found


if __name__ == "__main__":
    json_translation_dict = {
        "East Region": ["East"],
        "Industry": ["Industry Category"],
        " $100.0 - 499.9": [" $100.0 - $499.9"],
        "Gross Annual Revenue": ["Gross Revenue"],
        "FTEs": ["Full Time Employees"],
        " 1000 - 5000": [" 1,000.0 - 5,000.0"]
    }
    # define approved list
    approved_replacement_dict = json_translation_dict

    compd_datacuts_2017_list = [
        "East Region; Revenue 1 - 25",
        "East Region; Gross Annual Revenue: $100.0 - 499.9",
        "National; Industry: Healthcare",
        "East Region; Gross Annual Revenue: $100.0 - 499.9; FTEs: 1000 - 5000",
        "Florida; Miami Greater Area"
    ]
    # define old labels list
    old_list = compd_datacuts_2017_list

    compd_datacuts_2018_list = [
        "East Region; Revenue 1 - 25",
        "National; Industry Category: Healthcare",
        "East Region; Gross Revenue: $100.0 - $499.9",
        "East; Gross Revenue: $100.0 - $499.9; Full Time Employees: 1,000.0 - 5,000.0"
        ]
    # define new labels list
    new_list = compd_datacuts_2018_list

    close_compare = ReplacementListComparer(old_list, new_list, approved_replacement_dict)
    translation, not_found = close_compare.compare()
    print("Translation: {}".format(translation))
    print("Not found: {}".format(not_found))
import json
import multiprocessing

lock_last_result_2 = multiprocessing.Lock()

output_file_path = "results/res_config"
last_result = 0

def update_json_field_for_injection_tool(file_path, new_value_dmsc, new_value_2dnr, res_file_path):
    # Read the existing JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    for item in data.get("root", []):
        if ("classname" in item):
            classname = item.get("classname", [])
            if (classname == "ADmscv2"):
                item["tables"]["dmsc_sharpen_factor_black"] = [new_value_dmsc] * len(item["tables"]["gains"])
                item["tables"]["dmsc_sharpen_factor_white"] = [new_value_dmsc] * len(item["tables"]["gains"])
                item["tables"]["dmsc_sharpen_clip_black"] = [2*new_value_dmsc] * len(item["tables"]["gains"])
                item["tables"]["dmsc_sharpen_clip_white"] = [2*new_value_dmsc] * len(item["tables"]["gains"])
            if (classname == "A2dnrv5"):
                item["tables"]["sigma"] = [new_value_2dnr]* len(item["tables"]["gains"])

    with open(res_file_path, 'w') as file:
        json.dump(data, file, indent=4)


def update_json_fiels_for_rsim(file_path, values, keys, new_file_path, output_path, raw_path):
    # Load the JSON data
    with open(file_path, 'r') as f:
        with lock_last_result_2:
            data = json.load(f)
    # Update the dmsc_sharpen_factor_black value to the new desired value
            for root_item in data["root"]:
                # import ipdb; ipdb.set_trace()
                for driver in root_item["driver"]:
                    if driver.get("class") =="CDmscv2" or driver.get("class") == "C2dnrv3":
                        for i in range(len(keys)):
                            if (keys[i] in driver):
                                driver[keys[i]] = values[i]
                root_item["output"]["path"] = output_path
                root_item["input"]["path"] = [raw_path]
                        

            # Save the updated JSON data to the file
            with open(new_file_path, 'w') as file:
                json.dump(data, file, indent=4)



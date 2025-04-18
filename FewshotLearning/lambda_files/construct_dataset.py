import json
import csv

def load_data():
    """Loads and preprocesses data."""
    
    results = []
    
    with open("flipkart_com-ecommerce_sample.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        
        # Process each row
        for row in reader:
            # Apply filters
            category_match = row["product_category_tree"] == "[\"Jewellery >> Necklaces & Chains >> Necklaces\"]"
            description_length = len(row["description"]) < 300
            
            if category_match and description_length:
                # Process specifications
                spec_json = row["product_specifications"]
                spec_text = ""
                pairs = json.loads(spec_json.replace("=>", ":"))["product_specification"]
                
                for pair in pairs:
                    if pair["key"] not in ["Warranty Service Type", "Not Covered in Warranty", "Brand", "Name"]:
                        value = pair["value"]
                        if value == "The product is covered under 30 days Replacement Guarantee.":
                            value = "30 days replacement guarantee"
                        elif value == "Warranty Against Any Manufacturing Defect. The Warranty Is For 6 Months":
                            value = "6 Months Warranty"
                        
                        spec_text += pair["key"] + ": " + value + ", "
                
                spec_text = spec_text[:-2] if spec_text else ""
                clean_description = row["description"].replace("\\n", "").replace("\\t", "")
                
                result = {
                    "brand": row["brand"],
                    "product_name": row["product_name"],
                    "retail_price": row["retail_price"],
                    "discounted_price": row["discounted_price"],
                    "product_specifications": spec_text,
                    "description": clean_description
                }
                
                results.append(result)
    
    return results
    
from upload_processor import clean_extracted_text

sample = """
Wo rking definition
alter th e geometry
ass embly operation
machining opera tions
A station
A manufacturing
/G43/G48/G41/G50/G54/G45/G52/G20/G34
"""

print(clean_extracted_text(sample))
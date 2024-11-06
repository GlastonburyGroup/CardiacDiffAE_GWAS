import pandas as pd

cardiac_mappings = {
    "atherosclerotic": ["angina pectoris", "myocardial infarction", "coronary heart disease"],
    "heart failure": ["heart failure", "left ventricular failure", "cardiomegaly"],
    "conduction block": ["Tawara branches", "atrioventricular block", "fascicular block"],
    "metabolic syndrome": ["hypertension", "high cholesterol", "type 2 diabetes"],

    #Carlos (Cardiologist, Andrea) asked to ignore, because they are not precise enough.
    "miscellaneous": ["heart/cardiac problem", "other heart diseases"],

    #Defined by Soumick
    "extended miscellaneous": ["heart/cardiac problem", "other heart diseases", "thoracic aortic aneurysm", "myocarditis", "aneurysm of heart", "intracardiac thrombosis", "coronary artery aneurysm", "embolism and thrombosis of vena cava"],
    "arrythmias atrial/ventricular": ["arrhythmia/tachycardia", "premature depolarisation"],
    "extended arrythmias atrial/ventricular": ["arrhythmia/tachycardia", "premature depolarisation", "cardiac arrest", "atrial fibrillation/flutter"],

    #Was kept as individual ones by Carlos, even if the data is not sufficient.
    "hypertension": "hypertension",
    "cardiac arrest": "cardiac arrest",
    "aortic valve disorders": "aortic valve disorders",
    "mitral problems": "mitral problems",
    "atrial fibrillation/flutter": "atrial fibrillation/flutter",
    "pericardial problem": "pericardial problem",
    
    #The individual ones are also kepts as the data is sufficient [>10k subjects]
    "coronary heart disease": "coronary heart disease",
    "high cholesterol": "high cholesterol",    
    "angina pectoris": "angina pectoris",
    "myocardial infarction": "myocardial infarction",
    "type 2 diabetes": "type 2 diabetes",

    #Carlos (Cardiologist, Andrea) asked to ignore because they are tricky and it's hard to evaluate with diagnosis alone
    # "hypotension": "hypotension", #13k
    # "Raynauds syndrome": "Raynauds syndrome", #4k  

    # #Will be ignored as individual ones. Carlos asked to look for subtypes.
    # "premature depolarisation": "premature depolarisation",
    # "arrhythmia/tachycardia": "arrhythmia/tachycardia",
    # "cardiomyopathy": "cardiomyopathy",

    # #Will be ignored, as suggested by Carlos (Cardiologist, Andrea). Also, there are less than 1k subjects, so they won't pass our filter.
    # "thoracic aortic aneurysm": "thoracic aortic aneurysm",
    # "myocarditis": "myocarditis",
    # "aneurysm of heart": "aneurysm of heart",
    # "intracardiac thrombosis": "intracardiac thrombosis",
    # "coronary artery aneurysm": "coronary artery aneurysm",
    # "embolism and thrombosis of vena cava": "embolism and thrombosis of vena cava", 
}

def add_grouped_diseases(df, mappings, healthy_tag):
    if not bool(df.index.name):
        df.index.name = "eid"
    df_copy = df.copy()

    disease_to_groups = {}
    for group, diseases in mappings.items():
        if isinstance(diseases, list):
            for disease in diseases:
                if disease not in disease_to_groups:
                    disease_to_groups[disease] = []
                disease_to_groups[disease].append(group)
    mapping_df = pd.DataFrame([(disease, group) for disease, groups in disease_to_groups.items() for group in groups], columns=['summary', 'group'])

    df_copy.reset_index(inplace=True)
    merged_df = df_copy.merge(mapping_df, on='summary', how='left')
    grouped_df = merged_df.dropna(subset=['group']).copy()
    grouped_df['summary'] = grouped_df['group']
    grouped_df.drop(columns=['group'], inplace=True)
    grouped_df.set_index(df.index.names, inplace=True)
    print(f"Added {grouped_df['summary'].nunique()} grouped diseases.")

    df = pd.concat([df, grouped_df], ignore_index=False)
    print(f"Currently, there are {df['summary'].nunique()-1} diseases.")
    return df[df['summary'].isin(list(mappings.keys())+[healthy_tag])]
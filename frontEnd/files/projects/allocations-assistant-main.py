import os, json, difflib, re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from routers import hours

# LangChain imports
from langchain_community.llms import Ollama # Example for a local LLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# For combined retrievers
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.documents import Document as LC_Document

# Sentence transformer import
from sentence_transformers import SentenceTransformer

# For counting tokens
from transformers import AutoTokenizer

# BEGIN CODE FOR TAGGING USER INPUT
# --- Tag Hierarchy Mapping ---
# This dictionary maps a base tag to its parent and super category.
TAG_HIERARCHY = {
    "Aerodynamics": {"parent": "Aerospace_Aeronautical_and_Astronautical_Engineering", "super": "Engineering"},
    "Aerospace_engineering": {"parent": "Aerospace_Aeronautical_and_Astronautical_Engineering", "super": "Engineering"},
    "Space_Technology": {"parent": "Aerospace_Aeronautical_and_Astronautical_Engineering", "super": "Engineering"},
    "Biological_and_Biosystems_engineering": {"parent": "Bioengineering_and_Biomedical_Engineering", "super": "Engineering"},
    "Biomaterials_engineering": {"parent": "Bioengineering_and_Biomedical_Engineering", "super": "Engineering"},
    "Biomedical_technology": {"parent": "Bioengineering_and_Biomedical_Engineering", "super": "Engineering"},
    "Medical_engineering": {"parent": "Bioengineering_and_Biomedical_Engineering", "super": "Engineering"},
    "Biochemical_engineering": {"parent": "Chemical_Engineering", "super": "Engineering"},
    "Chemical_and_biomolecular_engineering": {"parent": "Chemical_Engineering", "super": "Engineering"},
    "engineering_chemistry": {"parent": "Chemical_Engineering", "super": "Engineering"},
    "Paper_science": {"parent": "Chemical_Engineering", "super": "Engineering"},
    "Petroleum_refining_process": {"parent": "Chemical_Engineering", "super": "Engineering"},
    "Polymer_plastics_engineering": {"parent": "Chemical_Engineering", "super": "Engineering"},
    "Architectural_engineering": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Construction_engineering": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Engineering_management_administration": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Environmental_environmental_health_engineering": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Geotechnical_and_geoenvironmental_engineering": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Sanitary_engineering": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Structural_engineering": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Surveying_engineering": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Transportation_and_highway_engineering": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Water_resource_engineering": {"parent": "Civil_Engineering", "super": "Engineering"},
    "Communication_engineering": {"parent": "Electrical_Electronic_and_Communications_Engineering", "super": "Engineering"},
    "Computer_engineering": {"parent": "Electrical_Electronic_and_Communications_Engineering", "super": "Engineering"},
    "Computer_hardware_engineering": {"parent": "Electrical_Electronic_and_Communications_Engineering", "super": "Engineering"},
    "Computer_software_engineering": {"parent": "Electrical_Electronic_and_Communications_Engineering", "super": "Engineering"},
    "Electrical_and_electronic_engineering": {"parent": "Electrical_Electronic_and_Communications_Engineering", "super": "Engineering"},
    "Laser_and_optical_engineering": {"parent": "Electrical_Electronic_and_Communications_Engineering", "super": "Engineering"},
    "Power": {"parent": "Electrical_Electronic_and_Communications_Engineering", "super": "Engineering"},
    "Telecommunications_engineering": {"parent": "Electrical_Electronic_and_Communications_Engineering", "super": "Engineering"},
    "Electromechanical_engineering": {"parent": "Mechanical_Engineering", "super": "Engineering"},
    "Mechatronics_robotics_and_automation_engineering": {"parent": "Mechanical_Engineering", "super": "Engineering"},
    "Industrial_engineering": {"parent": "Industrial_and_Manufacturing_Engineering", "super": "Engineering"},
    "Manufacturing_engineering": {"parent": "Industrial_and_Manufacturing_Engineering", "super": "Engineering"},
    "Operations_research": {"parent": "Industrial_and_Manufacturing_Engineering", "super": "Engineering"},
    "Systems_engineering": {"parent": "Industrial_and_Manufacturing_Engineering", "super": "Engineering"},
    "Ceramic_sciences_and_engineering": {"parent": "Metallurgical_and_Materials_Engineering", "super": "Engineering"},
    "Geophysical_geological_engineering": {"parent": "Metallurgical_and_Materials_Engineering", "super": "Engineering"},
    "Materials_Engineering": {"parent": "Metallurgical_and_Materials_Engineering", "super": "Engineering"},
    "Metallurgical_engineering": {"parent": "Metallurgical_and_Materials_Engineering", "super": "Engineering"},
    "Mining_and_mineral_engineering": {"parent": "Metallurgical_and_Materials_Engineering", "super": "Engineering"},
    "Textile_sciences_and_engineering": {"parent": "Metallurgical_and_Materials_Engineering", "super": "Engineering"},
    "Welding": {"parent": "Metallurgical_and_Materials_Engineering", "super": "Engineering"},
    "Agricultural_engineering": {"parent": "Other_Engineering", "super": "Engineering"},
    "Engineering_design": {"parent": "Other_Engineering", "super": "Engineering"},
    "Engineering_mechanics_physics_and_science": {"parent": "Other_Engineering", "super": "Engineering"},
    "Forest_engineering": {"parent": "Other_Engineering", "super": "Engineering"},
    "Nanotechnology": {"parent": "Other_Engineering", "super": "Engineering"},
    "Naval_architecture_and_marine_engineering": {"parent": "Other_Engineering", "super": "Engineering"},
    "Nuclear_engineering": {"parent": "Other_Engineering", "super": "Engineering"},
    "Ocean_engineering": {"parent": "Other_Engineering", "super": "Engineering"},
    "Petroleum_engineering": {"parent": "Other_Engineering", "super": "Engineering"},
    "Other_engineering_fields_that_cannot_be_classified_using_the_fields_above": {"parent": "Other_Engineering", "super": "Engineering"},
    "Astronomy": {"parent": "Astronomy_and_Astrophysics", "super": "Physical_Sciences"},
    "Astrophysics": {"parent": "Astronomy_and_Astrophysics", "super": "Physical_Sciences"},
    "Planetary_astronomy_and_science": {"parent": "Astronomy_and_Astrophysics", "super": "Physical_Sciences"},
    "Analytical_Chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Chemical_physics": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Environmental_chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Forensic_chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Inorganic_chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Organic_chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Organo-metallic_chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Physical_chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Polymer_chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Theoretical_chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Computational_chemistry": {"parent": "Chemistry", "super": "Physical_Sciences"},
    "Acoustics": {"parent": "Physics", "super": "Physical_Sciences"},
    "Atomic_molecular_physics": {"parent": "Physics", "super": "Physical_Sciences"},
    "Condensed_matter_and_materials_physics": {"parent": "Physics", "super": "Physical_Sciences"},
    "Elementary_particle_physics": {"parent": "Physics", "super": "Physical_Sciences"},
    "Mathematical_physics": {"parent": "Physics", "super": "Physical_Sciences"},
    "Nuclear_physics": {"parent": "Physics", "super": "Physical_Sciences"},
    "Optics_optical_sciences": {"parent": "Physics", "super": "Physical_Sciences"},
    "Plasma_high-temperature_physics": {"parent": "Physics", "super": "Physical_Sciences"},
    "Theoretical_physics": {"parent": "Physics", "super": "Physical_Sciences"},
    "Materials_chemistry": {"parent": "Materials_Science", "super": "Physical_Sciences"},
    "Materials_science": {"parent": "Materials_Science", "super": "Physical_Sciences"},
    "Other_physical_sciences_that_cannot_be_classified_using_the_fields_listed_above": {"parent": "Other_Physical_Sciences", "super": "Physical_Sciences"},
    "Aeronomy": {"parent": "Atmospheric_Sciences_and_Meteorology", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Atmospheric_chemistry_and_climatology": {"parent": "Atmospheric_Sciences_and_Meteorology", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Atmospheric_Physics_and_dynamics": {"parent": "Atmospheric_Sciences_and_Meteorology", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Extraterrestrial_atmospheres": {"parent": "Atmospheric_Sciences_and_Meteorology", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Meteorology": {"parent": "Atmospheric_Sciences_and_Meteorology", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Solar": {"parent": "Atmospheric_Sciences_and_Meteorology", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Weather_modification": {"parent": "Atmospheric_Sciences_and_Meteorology", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Earth_and_planetary_sciences": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Geochemistry": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Geodesy_and_gravity": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Geology": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Geomagnetism": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Geophysics_and_seismology": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Hydrology_and_water_resources": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Minerology_and_petrology": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Paleomagnetism": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Paleontology": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Physical_geography": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Stratigraphy_and_sedimentation": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Surveying": {"parent": "Geological_and_Earth_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Biological_oceanography": {"parent": "Ocean_Sciences_and_Marine_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Geological_oceanography": {"parent": "Ocean_Sciences_and_Marine_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Marine_biology": {"parent": "Ocean_Sciences_and_Marine_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Marine_oceanography": {"parent": "Ocean_Sciences_and_Marine_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Marine_sciences": {"parent": "Ocean_Sciences_and_Marine_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Oceanography_chemical_and_physical": {"parent": "Ocean_Sciences_and_Marine_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Other_related_disciplines_that_cannot_be_classified_using_the_fields_listed_above": {"parent": "Other_Geosciences_Atmospheric_and_Ocean_Sciences", "super": "Geosciences_Atmospheric_and_Ocean_Sciences"},
    "Applied_mathematics": {"parent": "Applied_mathematics", "super": "Mathematics_and_Statistics"},
    "Mathematics": {"parent": "Mathematics", "super": "Mathematics_and_Statistics"},
    "Statistics": {"parent": "Statistics", "super": "Mathematics_and_Statistics"},
    "Artificial_intelligence": {"parent": "Artificial_intelligence", "super": "Computer_and_Information_Sciences"},
    "Image_Processing": {"parent": "Image_Processing", "super": "Computer_and_Information_Sciences"},
    "Big_Data_Analytics": {"parent": "Big_Data_Analytics", "super": "Computer_and_Information_Sciences"},
    "Computer_Vision": {"parent": "Computer_Vision", "super": "Computer_and_Information_Sciences"},
    "Natural_Language_Processing": {"parent": "Natural_Language_Processing", "super": "Computer_and_Information_Sciences"},
    "Deep_Learning": {"parent": "Deep_Learning", "super": "Computer_and_Information_Sciences"},
    "Machine_Learning": {"parent": "Machine_Learning", "super": "Computer_and_Information_Sciences"},
    "Computer_and_information_technology_administration_and_management": {"parent": "Computer_and_information_technology_administration_and_management", "super": "Computer_and_Information_Sciences"},
    "Computer_science": {"parent": "Computer_science", "super": "Computer_and_Information_Sciences"},
    "Computer_software_and_media_applications": {"parent": "Computer_software_and_media_applications", "super": "Computer_and_Information_Sciences"},
    "Computer_systems_analysis": {"parent": "Computer_systems_analysis", "super": "Computer_and_Information_Sciences"},
    "Computer_systems_networking_and_telecommunications": {"parent": "Computer_systems_networking_and_telecommunications", "super": "Computer_and_Information_Sciences"},
    "Data_processing": {"parent": "Data_processing", "super": "Computer_and_Information_Sciences"},
    "Information_Sciences_studies": {"parent": "Information_Sciences_studies", "super": "Computer_and_Information_Sciences"},
    "Information_technology": {"parent": "Information_technology", "super": "Computer_and_Information_Sciences"},
    "Agricultural_business_and_management": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Agricultural_chemistry": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Agricultural_production_operations": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Animal_sciences": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Applied_horticulture_and_horticultural_business_services": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Aquaculture": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Fishing_and_fisheries_sciences_and_management": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Food_science_and_technology": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Forestry": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "International_agriculture": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Plant_sciences": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Soil_Sciences": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Veterinary_biomedical_and_clinical_sciences": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Veterinary_medicine": {"parent": "Agricultural_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Allergies_and_immunology": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Biochemistry_biophysics_and_molecular_biology": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Biogeography": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Biology_and_biomedical_sciences_general": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Biomathematics,_bioinformatics,_and_computational_biology": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Biotechnology": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Botany_and_plant_biology": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Cell_cellular_biology_and_anatomical_sciences": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Epidemiology_ecology_and_population_biology": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Foods_nutrition_and_wellness_studies": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Genetics": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Microbiological_sciences_and_immunology": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Molecular_medicine": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Neurobiology_and_neuroscience": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Pharmacology_and_toxicology": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Physiology_pathology_and_related_sciences": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Zoology_animal_biology": {"parent": "Biological_and_Biomedical_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Advance_graduate_dentistry_and_oral_sciences": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Allied_health_and_medical_assisting_services": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Bioethics_medical_ethics": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Clinical/medical_laboratory_science/research_and_allied_professions": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Clinical_medicine_research": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Communication_disorders_sciences_and_services": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Dentistry": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Dietetics_and_clinical_nutrition_services": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Gerontology_health_sciences": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Health_and_medical_administrative_services": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Health_medical_preparatory_programs": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Kinesiology_and_exercise_science": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Medical_clinical_science_graduate_medical_studies": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Medical_illustration_and_informatics": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Medicine": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Mental_health": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Nursing": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Optometry": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Osteopathic_medicine_osteopathy": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Pharmacy_pharmaceutical_sciences_and_administration": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Podiatric_medicine_podiatry": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Public_health": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Radiological_science": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Registered_nursing_nursing_administration_nursing_research_and_clinical_nursing": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Rehabilitation_and_therapeutic_professions": {"parent": "Health_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Natural_resources_management_and_policy": {"parent": "Natural_Resources_and_Conservation", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Renewable_natural_resources": {"parent": "Natural_Resources_and_Conservation", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Wildlife_and_wildlands_science_and_management": {"parent": "Natural_Resources_and_Conservation", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Other_life_sciences_that_cannot_be_classified_using_the_fields_listed_above": {"parent": "Other_Life_Sciences", "super": "Life_Sciences_and_Veterinary_Medicine"},
    "Clinical_Psychology": {"parent": "Clinical_Psychology", "super": "Psychology"},
    "Counseling_and_applied_psychology": {"parent": "Counseling_and_applied_psychology", "super": "Psychology"},
    "Human_development": {"parent": "Human_development", "super": "Psychology"},
    "Research_and_experimental_psychology": {"parent": "Research_and_experimental_psychology", "super": "Psychology"},
    "Cultural_anthropology": {"parent": "Anthropology", "super": "Social_Sciences"},
    "Medical_anthropology": {"parent": "Anthropology", "super": "Social_Sciences"},
    "Physical_and_biological_anthropology": {"parent": "Anthropology", "super": "Social_Sciences"},
    "Agricultural_economics": {"parent": "Economics", "super": "Social_Sciences"},
    "Applied_economics": {"parent": "Economics", "super": "Social_Sciences"},
    "Business_development": {"parent": "Economics", "super": "Social_Sciences"},
    "Development_economics_and_international_development": {"parent": "Economics", "super": "Social_Sciences"},
    "Econometrics_and_quantitative_economics": {"parent": "Economics", "super": "Social_Sciences"},
    "Industrial_economics": {"parent": "Economics", "super": "Social_Sciences"},
    "International_economics": {"parent": "Economics", "super": "Social_Sciences"},
    "Labor_economics": {"parent": "Economics", "super": "Social_Sciences"},
    "Managerial_economics": {"parent": "Economics", "super": "Social_Sciences"},
    "Natural_resources_economics": {"parent": "Economics", "super": "Social_Sciences"},
    "Public_finance_and_fiscal_policy": {"parent": "Economics", "super": "Social_Sciences"},
    "Comparative_government": {"parent": "Political_Science_and_Government", "super": "Social_Sciences"},
    "Government": {"parent": "Political_Science_and_Government", "super": "Social_Sciences"},
    "Legal_systems": {"parent": "Political_Science_and_Government", "super": "Social_Sciences"},
    "Political_economy": {"parent": "Political_Science_and_Government", "super": "Social_Sciences"},
    "Political_science": {"parent": "Political_Science_and_Government", "super": "Social_Sciences"},
    "Political_theory": {"parent": "Political_Science_and_Government", "super": "Social_Sciences"},
    "Comparative_and_historical_sociology": {"parent": "Sociology_Demography_and_Population_Studies", "super": "Social_Sciences"},
    "Complex_organizations": {"parent": "Sociology_Demography_and_Population_Studies", "super": "Social_Sciences"},
    "Cultural_and_social_structure": {"parent": "Sociology_Demography_and_Population_Studies", "super": "Social_Sciences"},
    "Demography_and_population_studies": {"parent": "Sociology_Demography_and_Population_Studies", "super": "Social_Sciences"},
    "Group_interactions": {"parent": "Sociology_Demography_and_Population_Studies", "super": "Social_Sciences"},
    "Rural_sociology": {"parent": "Sociology_Demography_and_Population_Studies", "super": "Social_Sciences"},
    "Social_problems_and_welfare_theory": {"parent": "Sociology_Demography_and_Population_Studies", "super": "Social_Sciences"},
    "Sociology": {"parent": "Sociology_Demography_and_Population_Studies", "super": "Social_Sciences"},
    "Archeology": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Area_ethnic_cultural_gender_and_group_studies": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Cartography": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "City_urban_community_and_regional_planning": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Criminal_science_and_corrections": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Criminology": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Geography": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Gerontology_social_sciences": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "History_including_history_and_philosophy_of_science_and_technology": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "International_relations_and_national_security_studies": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Linguistics": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Public_policy_analysis": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Regional_studies": {"parent": "Other_Social_Sciences", "super": "Social_Sciences"},
    "Use_this_category_for_R&D_that_involves_at_least_one_S&E_field_if_it_is_impossible_to_report_multidisciplinary_or_interdisciplinary_R&D_expenditures_in_specific_fields": {"parent": "Other_Sciences", "super": "Other_Sciences"},
    "Business_administration": {"parent": "Business_Management_and_Business_Administration", "super": "Non-S&E_Fields"},
    "Business_management": {"parent": "Business_Management_and_Business_Administration", "super": "Non-S&E_Fields"},
    "Business_managerial_economics": {"parent": "Business_Management_and_Business_Administration", "super": "Non-S&E_Fields"},
    "Management_information_systems_and_services": {"parent": "Business_Management_and_Business_Administration", "super": "Non-S&E_Fields"},
    "Marketing_management_and_research": {"parent": "Business_Management_and_Business_Administration", "super": "Non-S&E_Fields"},
    "Communication_and_media_studies": {"parent": "Communication_and_Communications_Technologies", "super": "Non-S&E_Fields"},
    "Communications_technologies": {"parent": "Communication_and_Communications_Technologies", "super": "Non-S&E_Fields"},
    "Journalism": {"parent": "Communication_and_Communications_Technologies", "super": "Non-S&E_Fields"},
    "Radio_television_and_digital_communication": {"parent": "Communication_and_Communications_Technologies", "super": "Non-S&E_Fields"},
    "Education_administration_and_supervision": {"parent": "Education", "super": "Non-S&E_Fields"},
    "Education_research": {"parent": "Education", "super": "Non-S&E_Fields"},
    "Teacher_education_specific_levels_and_methods": {"parent": "Education", "super": "Non-S&E_Fields"},
    "Teaching_fields": {"parent": "Education", "super": "Non-S&E_Fields"},
    "English_language_and_literature_letters": {"parent": "Humanities", "super": "Non-S&E_Fields"},
    "Foreign_languages_and_literatures": {"parent": "Humanities", "super": "Non-S&E_Fields"},
    "Humanities_general": {"parent": "Humanities", "super": "Non-S&E_Fields"},
    "Liberal_arts_and_sciences": {"parent": "Humanities", "super": "Non-S&E_Fields"},
    "Philosophy_and_religious_studies": {"parent": "Humanities", "super": "Non-S&E_Fields"},
    "Theology_and_religious_vocations": {"parent": "Humanities", "super": "Non-S&E_Fields"},
    "Law": {"parent": "Law", "super": "Non-S&E_Fields"},
    "Legal_Studies": {"parent": "Law", "super": "Non-S&E_Fields"},
    "Social_Work": {"parent": "Social_Work", "super": "Non-S&E_Fields"},
    "Drama_Theatre_arts_and_stagecraft": {"parent": "Visual_and_Performing_Arts", "super": "Non-S&E_Fields"},
    "Film_video_and_photographic_arts": {"parent": "Visual_and_Performing_Arts", "super": "Non-S&E_Fields"},
    "Fine_and_studio_arts": {"parent": "Visual_and_Performing_Arts", "super": "Non-S&E_Fields"},
    "Music": {"parent": "Visual_and_Performing_Arts", "super": "Non-S&E_Fields"},
    "Architecture": {"parent": "Other_Non-S&E_Fields", "super": "Non-S&E_Fields"},
    "Family_consumer_sciences_and_human_sciences": {"parent": "Other_Non-S&E_Fields", "super": "Non-S&E_Fields"},
    "Landscape_architecture": {"parent": "Other_Non-S&E_Fields", "super": "Non-S&E_Fields"},
    "Library_science": {"parent": "Other_Non-S&E_Fields", "super": "Non-S&E_Fields"},
    "Military_technology_and_applied_science": {"parent": "Other_Non-S&E_Fields", "super": "Non-S&E_Fields"},
    "Parks_sports_recreation_leisure_and_fitness": {"parent": "Other_Non-S&E_Fields", "super": "Non-S&E_Fields"},
    "Public_administration_and_public_affairs": {"parent": "Other_Non-S&E_Fields", "super": "Non-S&E_Fields"},
    "Other_non-S&E_fields_that_cannot_be_classified_using_the_fields_listed_above": {"parent": "Other_Non-S&E_Fields", "super": "Non-S&E_Fields"},
}

TAGGING_CONFIGS = {
    "auburn": {
        "tag_list_file": "./auburnTagging/auburnBaseTags.txt",
        "hierarchical": True,
        "hierarchy_map": TAG_HIERARCHY,
        "vectorstore_path": "./chroma_db_files_Auburn_Tags"
    },
    "fos": {
        "tag_list_file": "./trainingDocuments/fosList.txt",
        "hierarchical": False,
        "vectorstore_path": "./chroma_db_files_FOS_Tags"
    },
    "ai": {
        "tag_list_file": "./AITagging/finalAITagList.txt",
        "hierarchical": False,
        "vectorstore_path": "./chroma_db_files_AI_Tags"
    }
}
QUERY_TAGGING_SETUP = "auburn"



def load_approved_tags(file_path):
    """Loads the official list of tags into a set for fast lookups."""
    try:
        with open(file_path, 'r') as f:
            tags = {line.strip() for line in f if line.strip()}
        print(f"Successfully loaded {len(tags)} tags from {file_path}")
        return tags
    except FileNotFoundError:
        print(f"FATAL ERROR: The tag list file was not found at '{file_path}'.")
        print(f"Please make sure '{os.path.basename(file_path)}' is in the same directory as this script.")
        return None

def build_prompt_query_tag(abstract, tag_list_str):
    """Constructs the prompt for the AI with clear instructions."""
    return f"""
The following user input is part of a research support request.
Your task is to analyze the provided input and assign 3-5 relevant tags from the 'Allowed Tags' list.
You must return at least 3 tags. They ALL must be direct copies of the tags from the list.
DO NOT CREATE TAGS THAT ARE NOT ON THE LIST.

**Allowed Tags:**
{tag_list_str}

**Project Description:**
{abstract}

**Crucial Rules:**
1. Your output must be 3 to 5 tags chosen WORD FOR WORD from the 'Allowed Tags' list, this includes formatting such as commas and underscores.
2. Tags must be separated by semicolons (e.g., 'Tag One; Tag Two; Tag Three').
3. You should never propose your own tags.
4. Do not include any other text, explanations, or pleasantries.
5. Only return 'NA' if the abstract is completely empty or nonsensical.
6. You should always return at least 3 tags
7. All of the projects relate to computers in some way. Slightly decrease your tendency to use these tags  These include but are not limited to: Computer_science; Artificial_intelligence; Machine_Learning; Image_Processing; Computer_systems_analysis
Clarifications:
Biomathematics,_bioinformatics,_and_computational_biology is ONE tag.
Biochemistry_biophysics_and_molecular_biology is ONE tag.
Cell_cellular_biology_and_anatomical_sciences is ONE tag.
If you are in doubt whether a tag is multiple or one long tag, assume it is ONE tag.
"""

def tag_user_input(user_text: str, tagging_scheme: str = QUERY_TAGGING_SETUP):
    """
    Tags the user's input text using the AI model and hierarchical mapping.

    Args:
        user_text: A string of text (e.g., an abstract) to be tagged.

    Returns:
        A dictionary containing the base, parent, and super tags, or None if an error occurs.
    """
    print("--- Starting User Input Tagging ---")
    LLM_MODEL = "mistral-large:latest"
    config = TAGGING_CONFIGS.get(tagging_scheme)
    if not config:
        raise ValueError(f"Unknown tagging scheme: {tagging_scheme}")
    TAG_LIST_FILE = config["tag_list_file"]

    # 1. Load approved tags
    approved_tags_set = load_approved_tags(TAG_LIST_FILE)
    if approved_tags_set is None:
        return None
    tag_list_for_prompt = "\n".join(f"- {tag}" for tag in approved_tags_set)

    # 2. Initialize AI model
    print(f"Initializing AI model: {LLM_MODEL}...")
    try:
        llm = Ollama(model=LLM_MODEL)
    except Exception as e:
        print(f"ERROR: Could not initialize the Ollama model. Is it running? Details: {e}")
        return None

    # 3. Build prompt and get AI response
    prompt_str = build_prompt_query_tag(user_text, tag_list_for_prompt)
    tag_prompt = ChatPromptTemplate.from_template("{input}")
    print("Sending request to AI...")
    try:
        response = llm.invoke(prompt_str)
        if isinstance(response, dict) and "answer" in response:
            ai_output = response["answer"]
        else:
            ai_output = str(response)
        ai_output = ai_output.strip()
    except Exception as e:
        print(f"ERROR: Failed to get response from LLM. Details: {e}")
        return None

    print(f"AI Response: {ai_output}")

    # 4. Parse response and validate tags
    if ai_output.upper() == "NA":
        print("AI returned 'NA'. No tags will be assigned.")
        return "We couldn't tag your query. Please provide more information."


    raw_tags = [tag.strip() for tag in ai_output.split(';') if tag.strip()]

    validated_base_tags = []
    for tag in raw_tags:
        if tag in approved_tags_set:
            validated_base_tags.append(tag)
        else:
            # Try to find a close match from approved tags
            best_match = difflib.get_close_matches(tag, approved_tags_set, n=1, cutoff=0.8)
            if best_match:
                print(f"[SIMILARITY FIX] Replacing '{tag}' with '{best_match[0]}'")
                validated_base_tags.append(best_match[0])
            else:
                print(f"[UNMATCHED TAG] '{tag}' could not be validated or fixed.")

    # 5. Map to Parent and Super Tags
    parent_tags = []
    super_tags = []
    if config.get("hierarchical"):
        hierarchy_map = config["hierarchy_map"]
        for base_tag in validated_base_tags:
            formatted_tag = base_tag.replace(' ', '_')
            if formatted_tag in hierarchy_map:
                info = hierarchy_map[formatted_tag]
                parent_tags.append(info["parent"])
                super_tags.append(info["super"])
            else:
                print(f"HIERARCHY WARNING: No mapping for validated tag '{base_tag}'")

    # 6. Finalize and return the structured tags
    result = {
        "base_tags": validated_base_tags,
        "parent_tags": sorted(list(set(parent_tags))),
        "super_tags": sorted(list(set(super_tags)))
    }
    print("Tags:")
    print("  Base Tags:", validated_base_tags)
    print("  Parent Tags:", sorted(list(set(parent_tags))))
    print("  Super Tags:", sorted(list(set(super_tags))))    
    print("--- Tagging Complete ---")
    return result


def score_match(proposal, user_tags, tagging_scheme):
    LOW_PRIORITY_TAGS = {
        "Infrastructure and Instrumentation": 0.3,
        "Science and Engineering Education": 0.5,
        "Staff Activities (ACCESS or SP)": 0.8,
        "Training": 0.8
    }

    score = 0
    if tagging_scheme == "auburn":
        for tag in user_tags["base_tags"]:
            if tag in proposal.get("Auburn Base Tags", []):
                score += 5
        for tag in user_tags.get("parent_tags", []):
            if tag in proposal.get("auburn-parent-tags", []):
                score += 2
        for tag in user_tags.get("super_tags", []):
            if tag in proposal.get("auburn-super-tags", []):
                score += 1

    elif tagging_scheme == "fos":
        for tag in user_tags["base_tags"]:
            if tag in proposal.get("FOS Tags", []):
                score += 5

    elif tagging_scheme == "ai":
        for tag in user_tags["base_tags"]:
            if tag in proposal.get("ai-tags", []):
                score += 5
    else:
        raise ValueError(f"Unknown tagging scheme: {tagging_scheme}")

    fos_tags = proposal.get("FOS Tags", [])
    if any(low_tag in fos_tags for low_tag in LOW_PRIORITY_TAGS):
        score -= 500000
    return score


# END CODE FOR TAGGING USER INPUT


# Helper function for summarizing allocations 
def summarizer(proposals):
    LLM_MODEL = "mistral-large:latest" # Consider changing to the smaller model for less computational power required
    try:
        llm = Ollama(model=LLM_MODEL)
    except Exception as e:
        print(f"ERROR: Could not initialize the Ollama model. Is it running? Details: {e}")
        return None

    access_allocations = []
    all_resource_lines = []

    for p in proposals:
        resources = []
        for r in p.get("resources", []):
            name = r.get("resourceName", "Unknown Resource")
            allocation = r.get("allocation", "N/A")
            units = r.get("units", "")

            # Collect ACCESS Credits separately
            if name == "ACCESS Credits":
                access_allocations.append(allocation)

            # Avoid repeating unit if same as name
            if units and units != name:
                resource_str = f"{name}: {allocation} {units}"
            else:
                resource_str = f"{name}: {allocation}"

            resources.append(resource_str)

        all_resource_lines.append("\n".join(resources))

    if not access_allocations:
        return "No ACCESS Credit allocations found to summarize."

    allocations_str = ", ".join(str(a) for a in access_allocations)
    all_resources_str = "\n\n".join(all_resource_lines)

    prompt_str = (
        "You are a scientific assistant for someone deciding how many ACCESS Credits to request. "
        "Your task is to analyze the following ACCESS Credit allocations:\n"
        f"{allocations_str}\n\n"
        "Give the range of the values for ACCESS CREDITS (min to max), and 1 sentence description of the distribution. "
        "Then, give a 1 sentence description of the resources used overall. You should name the resources used specifically. "
        "For example, if 4 out of the 5 projects used SDSC Expanse, you could say 'A majority of the projects used "
        "SDSC Expanse, meaning you might want to look into using that as well.' "
        "Here is the full list of resources for each project:\n\n"
        f"{all_resources_str}"
    )

    try:
        response = llm(prompt_str)
        return response
    except Exception as e:
        print(f"ERROR: LLM call failed: {e}")
        return None




# Counting tokens - deepseek-r1:70b can handle 128k tokens
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


# --- RAG Chain Setup ---

# Ensure the vector store exists
active_config = TAGGING_CONFIGS[QUERY_TAGGING_SETUP]
VECTORSTORE_PATH = active_config["vectorstore_path"]
print("[DEBUG] Using vector store at:", os.path.abspath(VECTORSTORE_PATH))
if not os.path.exists(VECTORSTORE_PATH):
    raise RuntimeError(f"Vector store not found at '{VECTORSTORE_PATH}'. Please run `ingest.py` first.")

# Load the embedding model
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

# Custom embedding function for Chroma
class ChosenEmbeddingFunction:
    def __call__(self, texts):
        return model.encode(texts, convert_to_tensor=True).tolist()

    def embed_query(self, text):
        return self.__call__([text])[0]

embedding_function = ChosenEmbeddingFunction()
db = Chroma(persist_directory=VECTORSTORE_PATH, collection_name="access-projects", embedding_function=embedding_function)

''' Original way of finding documents
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 6,
        "filter": {
            "tags": {"$nin": ["Staff Activities (ACCESS or SP)", "Training"]}
        }
    }
)
'''


# --- Metadata-aware reranking setup ---

def metadata_reranker(results, priority_tags, LOW_PRIORITY_TAGS, w_semantic=0.6, w_penalty=0.2, w_boost=0.2):
    """
    Reranks a list of documents based on semantic score, metadata tags,
    and a defined weighting system.

    The original semantic score is normalized to a 0-1 range (where 1 is best)
    using the formula: similarity = 1 - (original_score / 2).

    The final adjusted score is a linear combination of:
    - Normalized semantic score (weighted by w_semantic)
    - Penalty for low-priority tags (weighted by w_penalty, subtracted)
    - Boost for priority tags (weighted by w_boost, added)

    Args:
        results (list): A list of tuples, where each tuple contains (document_object, original_semantic_score).
                        It's assumed original_semantic_score is a distance metric, typically in a 0-2 range,
                        where 0 is a perfect match and higher values indicate less similarity.
        priority_tags (list): A list of strings representing tags that should boost a document's score.
        LOW_PRIORITY_TAGS (dict): A dictionary where keys are low-priority tag strings and values are
                                  their corresponding penalty amounts.
        w_semantic (float): Weight for the normalized semantic score component. (Default: 0.6)
        w_penalty (float): Weight for the penalty score component. (Default: 0.2)
        w_boost (float): Weight for the boost (overlap) component. (Default: 0.2)

    Returns:
        list: A new list of tuples (document_object, adjusted_score), sorted in descending order
              by adjusted_score (higher adjusted_score means a better match).
    """
    reranked = []

    for doc, original_score in results:
        # 1. Normalize the original score to a 0-1 range, where higher is better.
        # This uses your specified equation: similarity = 1 - (score / 2)
        # We assume 'original_score' is a distance metric, ideally in a [0, 2] range,
        # where 0 is perfect similarity.
        normalized_semantic_score = 1 - (original_score / 2)

        # Ensure the normalized score stays within the [0, 1] bounds,
        # in case original_score goes slightly out of the expected [0, 2] range.
        normalized_semantic_score = max(0.0, min(1.0, normalized_semantic_score))

        # Extract and clean tags from document metadata
        tags = doc.metadata.get("tags", [])
        if isinstance(tags, str):
            # Defensive: if tags is a string, convert it to a list by splitting commas
            tags = [t.strip() for t in tags.split(",")]

        # Calculate the 'overlap' with priority_tags (this acts as a boost)
        overlap = len(set(tags) & set(priority_tags))

        # Calculate the 'penalty_score' from LOW_PRIORITY_TAGS
        # Each low-priority tag contributes its defined penalty value
        penalty_score = sum(LOW_PRIORITY_TAGS.get(tag, 0) for tag in tags)

        # 2. Apply the linear combination with explicit weights
        # The goal is that a higher 'adjusted_score' indicates a better match.
        # - Normalized semantic score contributes positively.
        # - Penalty score contributes negatively (higher penalty means lower adjusted score).
        # - Overlap (boost) contributes positively.
        adjusted_score = (w_semantic * normalized_semantic_score) \
                         - (w_penalty * penalty_score) \
                         + (w_boost * overlap)

        reranked.append((doc, adjusted_score))

    # Sort the reranked documents in descending order based on the adjusted_score.
    # A higher adjusted_score now means a better match.
    return sorted(reranked, key=lambda x: x[1], reverse=True)


def get_reranked_documents(query, priority_tags, LOW_PRIORITY_TAGS):
    # Use raw similarity search with score and filter
    raw_results = db.similarity_search_with_score(query, k=100)
    reranked = metadata_reranker(raw_results, priority_tags, LOW_PRIORITY_TAGS)
    return [doc for doc, _ in reranked[:5]]  # top 5

# Ensures that the database contains documents
print(f"[INFO] Vector store contains {len(db.get()['documents'])} documents.")

''' This is for retreiving reference documents supplementally
# Add a second retriever for reference documents
reference_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,
        "filter": {"tags": {"$in": ["Reference"]}}
    }
)
'''

# Add a second retriever for reference documents replaced by direct db.get call
raw_refs = db.get(where={"tags": {"$in": ["Reference"]}})
reference_docs = [
    LC_Document(page_content=doc, metadata=meta)
    for doc, meta in zip(raw_refs["documents"], raw_refs["metadatas"])
]

# Define the prompt template for the RAG chain
system_prompt = (
    "You are an intelligent assistant for ACCESS resource users to calculate how many ACCESS credits they should request. "
    "You have strict instructions to not make calculations to garner how many ACCESS credits they need, rather solely relying on "
    "precedence set by similar proposals. Start by identifying the past proposal's allocationType, and then base your answer on that. "
    "Decide which tier of ACCESS allocation should be given based off the tiers of the previous proposals that are sent to you. "
    "Then, give an estimate of how many credits would be necessary BASED OFF the PREVIOUS PROPOSALS. "
    "\n\n"
    "Your primary task is to find similar past proposals from the context and draw connections to the current question. "
    "You should pull DIRECTLY from these past proposals, without changing or hallucinating a single word. "
    "\n\n"
    "There are three ACCESS allocation tiers: Explore (400,000 ACCESS Credits), Discover (1,500,000 ACCESS Credits), and Accelerate (3,000,000 ACCESS Credits). "
    #"Read about them in the uploaded Resource files. The higher tiers require more work to be granted, so select the lowest possible tier "
    #"that is similar to other proposals in their field of research for their project\n\n"
    "You should return the following output with a brief explanation. "
    "At the end of your answer your findings should be displayed like this \n\n"
    "ACCESS Tier: ____\n"
    "ACCESS Credits: ____\n"
    "Possible resources to use: ____"
    "\n\n\n"
    "Similar previous title: \n"
    "Similar previous abstract: \n"
    "Similar previous resources: \n\n"
    "Then, you must print out the proposal titles that were passed to you as context, with a brief description of how it "
    "helped you make your allocation decision.\n"
    "USER QUERY:\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# --- Initialize the LLM ---
# This section shows how you would connect to a local LLM.
# You need a separate service running the model, like Ollama.
# For more info on Ollama: https://ollama.com/
#
# To run Llama 3 with Ollama (after installing Ollama):
# 1. Pull the model: `ollama pull llama3`
# 2. Run the server (this happens automatically in the background)
#
# If you don't have a local LLM, this will fail. You can replace this
# with another model provider if needed.
try:
    llm = Ollama(model="deepseek-r1:70b", temperature=0.3)
    print("Successfully connected to local Llama 3 model via Ollama.")
except Exception as e:
    print(f"Could not connect to local LLM: {e}")
    print("Please ensure you have a local LLM server (like Ollama) running and the model is available.")
    # As a fallback for demonstration, we will exit if no LLM is available.
    exit()


# Create the document combination chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the final retrieval chain - this uses a single retriever
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Custom retrieval chain that merges both sets of documents
def combined_retriever(input, tagArray):
    query = input["input"]

    # Flatten tagArray into a single list of unique tags
    if isinstance(tagArray, dict):
        priority_tags = list(set(
            tag for tag_list in tagArray.values() for tag in tag_list
        ))
    else:
        priority_tags = tagArray  # assume it's already a list

    # Penalization - The higher the decimal, the higher the penalty
    LOW_PRIORITY_TAGS = {
        "Infrastructure and Instrumentation": 0.3,
        "Science and Engineering Education": 0.8,
        "Staff Activities (ACCESS or SP)": 0.8,
        "Training": 0.8,
        "Uncategorized": 0.8
    }
    # Get relevant documents
    docs_general = get_reranked_documents(query, priority_tags, LOW_PRIORITY_TAGS)
    # Merge general + reference documents
    all_docs = {doc.page_content: doc for doc in (docs_general + reference_docs)}
    combined_docs = list(all_docs.values())
    return combined_docs


rag_chain = RunnableMap({
    "context": RunnableLambda(lambda x: combined_retriever(x, x["tagArray"])),
    "input": lambda x: x["input"]
}) | question_answer_chain


# Cleaning the final deepseek output
def clean_ai_output(text: str) -> str:
    # 1. Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Remove all HTML-like tags like <tag> or </tag>
    text = re.sub(r"</?\w+?>", "", text)

    # 3. Remove LaTeX-style delimiters
    text = re.sub(r"\\\(|\\\)", "", text)
    text = re.sub(r"\\\[|\\\]", "", text)

    # 4. Replace \text{...} with just the content
    text = re.sub(r"\\text\s*{([^}]*)}", r"\1", text)

    # 5. Replace common LaTeX symbols with readable equivalents
    replacements = {
        r"\\times": "×",
        r"\\cdot": "·",
        r"\\approx": "≈",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    # 6. Remove stray backslashes
    text = re.sub(r"\\", "", text)

    # 7. Strip extra spaces on each line (but keep line breaks)
    lines = text.splitlines()
    cleaned_lines = [re.sub(r" +", " ", line).strip() for line in lines]
    cleaned_text = "\n".join(cleaned_lines)

    return cleaned_text.strip()


# --- FastAPI Application ---

app = FastAPI(title="ACCESS RAG Chatbot")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

class ChatRequest(BaseModel):
    message: str
    
app.include_router(hours.router)


@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Serves the main HTML page."""
    with open("app/static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/chat")
async def handle_chat(chat_request: ChatRequest):
    """Handles chatbot interaction using the RAG chain."""
    try:
        
        # This is to count the tokens of the query
        query = chat_request.message

        # Tagging user input
        # Step 1: Tag the user input
        tagging_result = tag_user_input(query)

        if isinstance(tagging_result, str):  # error message returned
            return {"response": tagging_result}
        
        if tagging_result is None:
            return {"response": "Sorry, I couldn't tag your input. Please try again."}

        base_tags = tagging_result["base_tags"]
        parent_tags = tagging_result["parent_tags"]
        super_tags = tagging_result["super_tags"]
        # Load proposals (once, or stream through the file)
        proposals = []
        with open("./varyingTaggedJSONS/indexed_proposals.jsonl") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):  # ✅ skip empty or commented lines
                    continue
                try:
                    proposals.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse line {i}: {e}")
        # Score and sort
        scored = sorted(
            proposals,
            key=lambda p: score_match(p, tagging_result, QUERY_TAGGING_SETUP),
            reverse=True
        )
        # Take top 3–5 matches
        top_matches = scored[:5]
        proposals = top_matches

        docs = combined_retriever({"input": query}, tagging_result)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        print("=== Retrieved Documents ===")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(doc.page_content[:1000])  # First 1000 characters
            print("Metadata:", doc.metadata)
        full_input_text = system_prompt.format(context=context_text) + query
        token_count = count_tokens(full_input_text)
        print("=== Prompt Preview ===")
        print(full_input_text[:1000])  # just the beginning
        print("=== Total tokens:", token_count)

        # NOTE: The response from the local LLM might not be a dictionary,
        # but a direct string. Adjust parsing as needed.
        # Prepare context to estimate token usage
        response_text = rag_chain.invoke({
            "input": query,
            "tagArray": tagging_result  # this is your dict of base/parent/super tags
        })
        # The structure of the response can vary. We check if it's a dict with 'answer'.
        if isinstance(response_text, dict) and 'answer' in response_text:
            answer = response_text.get("answer", "I'm sorry, I couldn't find an answer.")
        else:
            # If it's not a dict or doesn't have 'answer', we assume the response is the text itself.
            ai_answer = str(response_text)
        # Turn proposals into readable text
        proposal_texts = ""
        for i, p in enumerate(proposals, 1):
            resource_lines = ""
            for r in p.get("resources", []):
                name = r.get("resourceName", "Unknown Resource")
                allocation = r.get("allocation", "N/A")
                units = r.get("units", "")

                # Avoid repeating units if they're the same as the resource name
                if units and units != name:
                    resource_lines += f"{name}: {allocation} {units}\n"
                else:
                    resource_lines += f"{name}: {allocation}\n"
            proposal_texts += (
                f"\nProposal {i}:\n"
                f"Title: {p.get('requestTitle', '')}\n"
                f"Abstract: {p.get('abstract', '')}\n"
                f"Resources:\n{resource_lines}"
            )
        summary_text = summarizer(proposals)

        # Strip <think>...</think> content from ai_answer
        ai_answer_clean = clean_ai_output(ai_answer)

        # Combine proposals and AI output into one response string
        full_response = (
            proposal_texts
            + "\n\n--- ALLOCATION SUMMARY ---\n\n"
            + summary_text
            + "\n\n--- AI RESPONSE ---\n\n"
            + ai_answer_clean
        )

        return {
            "proposals": proposal_texts,
            "summary": summary_text,
            "ai_response": ai_answer_clean
        }
    except Exception as e:
        print(f"Error during chat: {e}")
        return {"response": "An error occurred while processing your request."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7515)

from tika import parser
import re
import spacy
from spacy.matcher import Matcher
from PyPDF2 import PdfReader
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
from resume_matcher.labels import *

def calculate_cosine_similarity(job_skills, resume_skills):
    skills = [job_skills, resume_skills]

    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the skills into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(skills)

    # Calculate cosine similarity between the two TF-IDF vectors
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    # The result is a 2D array, accessing the similarity value
    similarity = cosine_sim[0][0]
    return similarity


def unique_values_to_string(arr):
    unique_values = list(set(arr))  # Convert the array to a set to get unique values, then convert back to a list
    unique_string = ', '.join(str(val) for val in unique_values)  # Join the unique values using a comma and space
    return unique_string

def get_email_addresses(string):
    r = re.compile(r"[\w\.-]+@[\w\.-]+")
    return unique_values_to_string(r.findall(string))

def get_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', num) for num in phone_numbers]

def extract_name(text):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    nlp_text = nlp(text)
  
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    
    matcher.add('NAME', [pattern], on_match = None)
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
       span = nlp_text[start:end]
       return span.text

def remove_symbols(text):
    # Remove symbols except letters, numbers, and spaces
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

def handleResume(resume):
    Keywords = ["education",
            "summary",
            "accomplishments",
            "executive profile",
            "professional profile",
            "personal profile",
            "work background",
            "academic profile",
            "other activities",
            "qualifications",
            "experience",
            "interests",
            "skills",
            "achievements",
            "publications",
            "publication",
            "certifications",
            "workshops",
            "projects",
            "internships",
            "trainings",
            "hobbies",
            "overview",
            "objective",
            "position of responsibility",
            "jobs"
           ]
    file_data = parser.from_file(settings.MEDIA_ROOT + resume)
    text = file_data["content"]
    parsed_content = {}
    email = get_email_addresses(text)
    parsed_content["E-mail"] = email
    phone_number= get_phone_numbers(text)
    if len(phone_number) <= 10:
        parsed_content['Phone number'] = phone_number

    parsed_content['Phone number'] = unique_values_to_string(parsed_content['Phone number'])
    name = extract_name(text)
    parsed_content['Name'] =  name
    text = text.replace("\n"," ")
    text = text.replace("[^a-zA-Z0-9]", " ");  
    re.sub('\W+','', text)
    text = text.lower()

    content = {}
    indices = []
    keys = []
    for key in Keywords:
        try:
            content[key] = text[text.index(key) + len(key):]
            indices.append(text.index(key))
            keys.append(key)
        except:
            pass
    zipped_lists = zip(indices, keys)
    sorted_pairs = sorted(zipped_lists)
    
    tuples = zip(*sorted_pairs)
    indices, keys = [ list(tuple) for tuple in  tuples]
    #Keeping the required content and removing the redundant part
    content = []
    for idx in range(len(indices)):
        if idx != len(indices)-1:
            content.append(text[indices[idx]: indices[idx+1]])
        else:
            content.append(text[indices[idx]: ])
    for i in range(len(indices)):
        parsed_content[keys[i]] = remove_symbols(content[i])
    return parsed_content

def get_recommended_jobs(category, skills):
    target_values = relations[category]

    similarity_ids = []

    # Read the CSV file and filter rows based on Role Category values
    with open('/home/prazzwalthapa/Desktop/NLP_Project_Jobs/Application/backend/matcher_api/resume_matcher/sample_jobs.csv', mode='r') as file:
        reader = csv.DictReader(file)
        filtered_rows = [row for row in reader if row.get('Role Category') in target_values]
    
    for job in filtered_rows:
        similarity = calculate_cosine_similarity(job['Key Skills'].replace('|', ','), skills)
        data = {
            'job': job,
            'similarity': similarity
        }
        if len(similarity_ids) < 5:
            similarity_ids.append(data)
        else:
            low = min(similarity_ids, key=lambda x: x['similarity'])
            if similarity > low['similarity']:
                similarity_ids.append(data)
                similarity_ids.remove(low)


    return similarity_ids
    

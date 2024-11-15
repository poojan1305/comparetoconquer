from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse

# Models and Forms
from .models import Urls, Contact

# Standard Library
import os
import csv
from io import BytesIO

# Third-party Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# For Azure OpenAI integration
from langchain_openai import AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

os.environ['AZURE_OPENAI_API_KEY']="9zjg7qdEiNjUeC96NRHHxn27NHzF5TGxO1UOmeRlShcbbIkmPgJDJQQJ99AKAC77bzfXJ3w3AAABACOGjlo4"
os.environ['AZURE_OPENAI_ENDPOINT']="https://genai-openai-comparetoconquer.openai.azure.com"
os.environ['OPENAI_API_VERSION']="2023-12-01-preview"

# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(azure_deployment='gpt-4o', temperature=0.8)




# Create your views here.


def home(request):
    return render(request, 'dashboard/home.html')





def success(request):
    return render(request, "dashboard/success.html")


def contact(request):
    if request.method == "POST":
        contact = Contact()
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')

        contact.name = name
        contact.email = email
        contact.message = message
        contact.save()

        return render(request, 'dashboard/success.html')
    return render(request, 'dashboard/contactus.html')















def about(request):
    

    return render(request, 'dashboard/aboutus.html')
   
def faq(request):
    return render(request,'dashboard/faq.html')


def about(request):
    

    return render(request, 'dashboard/aboutus.html')



def visualize_data(request):
    # Load data from CSV file
    file_path = 'visual.csv'  # Update with your actual CSV file path
    df = pd.read_csv(file_path)

    # Aggregate complaint counts by team
    team_complaints = df['team'].value_counts()

    # List to store plot images
    images = []

    # 1. Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x=team_complaints.values, y=team_complaints.index, palette='viridis')
    plt.title('Complaints Count by Team')
    plt.xlabel('Complaint Count')
    plt.ylabel('Team')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    plt.close()

    # 2. Pie Chart
    plt.figure(figsize=(6, 6))
    plt.pie(team_complaints, labels=team_complaints.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(team_complaints)))
    plt.title('Complaint Distribution by Team')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    plt.close()

    # 3. Word Cloud
    complaint_text = " ".join(df['team'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(complaint_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Complaints by Team')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    plt.close()

    # 4. Count Plot
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='team', palette='viridis', order=team_complaints.index)
    plt.title('Count Plot of Complaints by Team')
    plt.xlabel('Team')
    plt.ylabel('Complaint Count')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    plt.close()

    # Pass all images to the template
    return render(request, 'dashboard/visualization.html', {'images': images})


def qa_bot_view(request):
    result = None

    if request.method == "POST" and request.FILES.get("pdf_file"):
        uploaded_file = request.FILES["pdf_file"]
        file_path = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
        document_path = default_storage.path(file_path)

        # Initialize AzureChatOpenAI
        llm = AzureChatOpenAI(azure_deployment='gpt-4o', temperature=0.8)

        # Load and process PDF
        loader = PyPDFLoader(document_path)
        documents = loader.load()[:10]
        text_splitter = CharacterTextSplitter(chunk_size=12000)
        split_documents = text_splitter.split_documents(documents)

        # Embedding and vector store setup
        embedding = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        vector_store = FAISS.from_documents(split_documents, embedding)
        retriever = vector_store.as_retriever()

        # Define prompt template
        test_template = """ 
        Use the following pieces of context to answer the question at the end.
        If the question is not related to PDF, say that it is not related to PDF, don't try to make up an answer.
        Understand the table values also.
        PDF Context: {context}
        Question : {question}
        Helpful Answer:
        """
        qa_chain_prompt = PromptTemplate(input_variables=['context', 'question'], template=test_template)

        # Process question and generate answer
        question = request.POST.get("question")
        if question:
            context = retriever.invoke(question)
            prompt = qa_chain_prompt.invoke({'question': question, 'context': context})
            result = llm.invoke(prompt).content

    return render(request, "dashboard/qa_bot.html", {"result": result})


keywords = {
    'Credit Team': ['credit', 'report', 'reporting'],
    'Debt Team': ['debt', 'collection'],
    'Information Team': ['information', 'incorrect', 'account'],
    'Breach Support': ['improper', 'use', 'privacy'],
    'Technical Team': ['technical', 'issue', 'problem'],
    'Support Team': ['support', 'inquiry', 'question']
}

def gen_category(complaint_raised):
    temp_dict = {'Complaint': complaint_raised}
    questions = ["Complaint Category", "Company", "Problem", "Summary", "Pin Code", "City"]

    for question in questions:
        ask = f"Given statement is complaint logged by customer, please use this complain statement and answer questions asked in max 3 words one by one '{complaint_raised}' {question}"
        message = [("system", "You are a helpful assistant"), ("human", ask)]
        result_cat = llm.invoke(message)
        temp_dict[question] = result_cat.content

    return pd.DataFrame(temp_dict, index=[0])

def map_to_teams(issue_category):
    issue_category = issue_category.lower()
    for team, team_keywords in keywords.items():
        if any(keyword in issue_category for keyword in team_keywords):
            return team
    return 'Other team'

output_file_path ="visual.csv"

def process_complaint(request):
    # result = {}
    # if request.method == "POST":
    #     complaint_text = request.POST.get("complaint")
    #     if complaint_text:
    #         cat_df = gen_category(complaint_text)
    #         cat_df['team'] = cat_df.apply(lambda row: map_to_teams(row['Complaint Category']), axis=1)
    #         complaint_id = 1000  # Set a starting complaint number, this could be incremented as needed
    #         result = {
    #             "complaint_number": complaint_id,
    #             "assigned_team": cat_df.iloc[0]['team']
    #         }
    
    # return render(request, "dashboard/process_complaint.html", {"result": result})

    result = {}
    if request.method == "POST":
        complaint_text = request.POST.get("complaint")
        if complaint_text:
            # Generate complaint details
            cat_df = gen_category(complaint_text)
            cat_df['team'] = cat_df.apply(lambda row: map_to_teams(row['Complaint Category']), axis=1)

            # Ensure output_file_path points to a .csv file
            # output_file_path = 'path/to/output_file.csv'  # Ensure this is .csv

            # Load existing CSV file
            old_df = pd.read_csv(output_file_path)
            # Generate a new Complaint Number based on the row count
            complaint_id = old_df.shape[0] + 1
            cat_df['Compaint_No'] = complaint_id  # Add complaint number to new complaint

            # Append new complaint to the old DataFrame
            final_df = pd.concat([old_df, cat_df], ignore_index=True)
            # Save updated DataFrame back to CSV
            final_df.to_csv(output_file_path, index=False)

            # Return Complaint Number and Assigned Team to the template
            result = {
                "complaint_number": complaint_id,
                "assigned_team": cat_df.iloc[0]['team']
            }
    
    return render(request, "dashboard/process_complaint.html", {"result": result})


def csv_to_table(request):
    # Path to your CSV file
    csv_file_path = 'visual.csv'
    
    # Open and read the CSV file
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]
    
    # Pass the CSV data to the template
    return render(request, 'dashboard/tabular.html', {'rows': rows})






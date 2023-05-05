from google.cloud import storage
from pathlib import Path
import datetime
import os
import io
from tqdm import tqdm
import uuid
from google.cloud import storage
from google.cloud import bigquery
from google.resumable_media.requests import ResumableUpload
from google.auth.transport.requests import AuthorizedSession
from credentials import credentials

# bq functions obtained from https://blog.coupler.io/how-to-crud-bigquery-with-python/

def query_datasets_to_df(QUERY):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    
    query_job = client.query(QUERY)
    
    df = query_job.to_dataframe()
    return df

def query_datasets(QUERY):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    
    query_job = client.query(QUERY)
    rows = query_job.result()
    return rows

def list_tables(dataset_id):    
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    
    tables = client.list_tables(dataset_id)  
    print("Tables contained in '{}':".format(dataset_id))
    for table in tables:
        print("{}.{}.{}".format(table.project, table.dataset_id, table.table_id))

def delete_table(table_id):    
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    
    client.delete_table(table_id, not_found_ok=True)
    
    # google.api_core.exceptions.NotFound unless not_found_ok is True.
    print("Deleted table '{}'.".format(table_id))
    
def insert_df(df_to_insert, table_id):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    
    # Generate unique IDs for each row using UUID4
    df_to_insert['id'] = [str(uuid.uuid4()) for _ in range(len(df_to_insert))]

    errors = client.load_table_from_dataframe(df_to_insert, table_id)
    if errors == []:
        print(f"New dataframe have been added with len {len(df_to_insert)}")
    else:
        print(f"Encountered errors while inserting dataframe: {errors}")
        
def append_df_to_bq_table(df_to_insert, table_id):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
  
    query = f"SELECT img_url FROM `{table_id}`"
    rows = query_datasets(query)
    existing_rows = [x['img_url'] for x in list(rows)]
    print(f'Found {len(existing_rows)} in {table_id}')
    new_rows = df_to_insert[~df_to_insert['img_url'].isin(existing_rows)]
    print(f'Total new rows to append {len(new_rows)}')
    
    if len(new_rows)>0:
        # Generate unique IDs for each row using UUID4
        new_rows['id'] = [str(uuid.uuid4()) for _ in range(len(new_rows))]
        print("Generated id for new rows")
        
        errors = client.load_table_from_dataframe(new_rows, table_id)
        if errors == []:
            print(f"New dataframe have been added with len {len(new_rows)}")
        else:
            print(f"Encountered errors while inserting dataframe: {errors}")
    else:
        print('No new data to append')
            
def insert_rows(table_id, rows_to_insert):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
  
    errors = client.insert_rows_json(table_id, rows_to_insert)  # Make an API request.
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))

def create_table(dataset_id, table_name, schema):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    table_id = bigquery.Table.from_string(f"{dataset_id}.{table_name}")

    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table)
    print(
        "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
    )
    
def delete_dataset(dataset_id):
    """
    The delete_contents
    parameter is set as True
    to delete the dataset and its contents. Also, the not_found_ok
    is set as True
    to avoid raising an error if the dataset has already been deleted or is not found in the project."""
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    client.delete_dataset(
        dataset_id, delete_contents=True, not_found_ok=True
    )
    print(
        "Deleted dataset '{}'.".format(
            dataset_id
        )
    )


def update_dataset(dataset_id, new_description):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    dataset = client.get_dataset(dataset_id)
    dataset.description = new_description
    dataset = client.update_dataset(dataset, ["description"])
    full_dataset_id = "{}.{}".format(dataset.project, dataset.dataset_id)
    print(
        "Updated dataset '{}' with description '{}'.".format(
            full_dataset_id, dataset.description
        )
    )
    
def get_dataset(dataset_id):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    dataset = client.get_dataset(dataset_id)  # Make an API request.
    full_dataset_id = "{}.{}".format(dataset.project, dataset.dataset_id)
    friendly_name = dataset.friendly_name
    print(
        "Got dataset '{}' with friendly_name '{}'.".format(
            full_dataset_id, friendly_name
        )
    )
    
def list_datasets():
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    datasets = list(client.list_datasets())  # Make an API request.
    project = client.project
    if datasets:
        print("Datasets in project {}:".format(project))
        for dataset in datasets:
            print("\t{}".format(dataset.dataset_id))
    else:
        print("{} project does not contain any datasets.".format(project))
        
def create_dataset(project_name, dataset_name):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials)
    
    dataset_id = f"{project_name}.{dataset_name}"
    dataset_ref = bigquery.DatasetReference.from_string(dataset_id, default_project=client.project)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    dataset = client.create_dataset(dataset)
    
    print("Created dataset {}.{}".format(client.project, dataset.dataset_id))

def implicit():

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client(credentials=credentials)

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)


def create_bucket_class_location(bucket_name):
    """Create a new bucket in specific location with storage class"""
    # bucket_name = "your-new-bucket-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = "COLDLINE"
    new_bucket = storage_client.create_bucket(bucket, location="us")

    print(
        "Created bucket {} in {} with storage class {}".format(
            new_bucket.name, new_bucket.location, new_bucket.storage_class
        )
    )
    return new_bucket

def upload_blob(bucket_name, source_file_name, destination_blob_name, fileType):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    #storage_client = storage.Client.from_service_account_json(str(path_json))
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name, content_type=fileType, timeout=1000)

    # print(
    #     "File {} uploaded to {}.".format(
    #         source_file_name, destination_blob_name
    #     )
    # )
    

def is_stored(blob_path, bucket_name):
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    stats = storage.Blob(bucket=bucket, name=blob_path).exists(storage_client)
    return stats

def generate_image_url(blob_path, bucket_name):
    """ generate signed URL of a video stored on google storage. 
        Valid for 300 seconds in this case. You can increase this 
        time as per your requirement. 
    """         
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path) 
    return blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')

    
def retry_on_connectionerror(f, max_retries=5):
  retries = 0
  while retries < max_retries:
    try:
      return f()
    except ConnectionError:
      retries += 1
  raise Exception("Maximum retries exceeded")


def upload_file_to_gstore(bucket_name, path_file, file_name, fileType, chunk_size=1024 * 1024):
    
    authed_session = AuthorizedSession(credentials)
    data = open(path_file, 'rb').read()

    url_template = (
             f'https://www.googleapis.com/upload/storage/v1/b/{bucket_name}/o?'
             u'uploadType=resumable')

    upload_url = url_template.format(bucket=bucket_name)

    upload = ResumableUpload(upload_url, chunk_size)
    stream = io.BytesIO(data)

    metadata = {u'name': file_name}
    response = upload.initiate(authed_session, stream, metadata, fileType)

    upload_id = response.headers[u'X-GUploader-UploadID']
    
    print(f'File size: {upload.total_bytes} bytes')
    pbar = tqdm(total=upload.total_bytes)
    finished = upload.finished
    while not finished:
        response = upload.transmit_next_chunk(authed_session)
        pbar.update(upload.bytes_uploaded)
        finished = upload.finished
        # print(finished)
    pbar.close()
    print('File uploaded')


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
    
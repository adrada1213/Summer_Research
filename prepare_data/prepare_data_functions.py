import logging
import os
import numpy as np
from time import time
'''
import libraries to send email
'''
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

logger = logging.getLogger(__name__)

'''
This function gets the pointer paths of the patients from the cim directory
Input:  cim_dir = directory of the cim we want to get paths to the image pointers for

Output: ptr_files = paths to the image pointer files
'''
def get_pointer_paths(cim_dir):
    patient_folders = os.listdir(cim_dir)
    ptr_files = []
    for patient_name in patient_folders:
        system_dir = os.path.join(cim_dir, patient_name, "system")
        try:
            files = [f for f in os.listdir(system_dir) if f.endswith(".img_imageptr")]
        except FileNotFoundError:
            logger.error("The system cannot find the path specified {}".format(system_dir), exc_info=True)
            continue
        try:
            ptr_files.append(os.path.join(system_dir, files[0]))
        except IndexError:
            logger.error("No image pointer file for {} | CIM dir: {}".format(patient_name, os.path.join(cim_dir, patient_name)), exc_info=True)
            continue
        files = []
    
    return ptr_files

'''
This function calculates the time elapsed
Input:  start = start time
Output: hrs, mins, secs = time elapsed in hours, minutes, seconds format
'''
def calculate_time_elapsed(start):
    end = time()
    hrs = (end-start)//60//60
    mins = ((end-start) - hrs*60*60)//60
    secs = int((end-start) - mins*60 - hrs*60*60)

    return hrs, mins, secs

'''
This function logs info level messages and prints them
Input:  output_messages = string/strings(as list) of messages
'''
def log_and_print(output_messages):
    if isinstance(output_messages, str):    #if output message is a single string
        logger.info(output_messages)
        print(output_messages)
    else:    
        for message in output_messages:
            logger.info(message)
            print(message)

'''
This function logs error level messages and prints them
Input:  output_messages = string/strings(as list) of messages
'''
def log_error_and_print(output_messages):
    if isinstance(output_messages, str):    #if output message is a single string
        logger.error(output_messages)
        print(output_messages)
    else:    
        for message in output_messages:
            logger.error(message)
            print(message)

'''
This function gets the cim path of the patient
Inputs:  patient_name = name of patient from dicom header (with underscore)
        cim_patients = paths to the cim patients folder

Output: cim_path = returns the path to the cim (Format MODEL/CIM_PATIENT_NAME)
'''
def get_cim_path(patient_name, cim_patients):
    if patient_name != "4J_Y5_B5__XN":  #unique case where this patient has two underscores
        cim_ptr_path = [p for p in cim_patients if patient_name.replace("_Bio", "").lower() in p.lower()][0] #get the cim path of current patient
        cim_pat_name = os.path.basename(cim_ptr_path)
        cim_model_name = os.path.basename(os.path.dirname(cim_ptr_path))
        cim_path = "{}\\{}".format(cim_model_name, cim_pat_name)
    else:    
        cim_path = "CIM_DATA_ze_2\\4J_Y5_B5_XN_ze"

    return cim_path

'''
This functions puts all the paths of the cim patients in a list
Inputs:  cim_dir = directory to the cim models
        cim_models = name of the folders of the cim models

Output: cim_patients = list of all cim patient paths 
'''
def get_cim_patients(cim_dir, cim_models):
    # put the paths of the models in a list
    cim_models_paths = [os.path.join(cim_dir, d) for d in os.listdir(cim_dir) if d in cim_models]
    cim_patients = []
    # obtain the cim path of each patient
    for cim_model in cim_models_paths:
        cim_patients += [os.path.join(cim_model, d) for d in os.listdir(cim_model) if os.path.isdir(os.path.join(cim_model, d))]
    
    return cim_patients

'''
This function gets the slices (e.g. 0 1 2) from a pointer
Input:  ptr_content = content of the current pointer (can read using np.genfromtxt)

Output: slices = the slices as a np.array 
'''
def get_slices(ptr_content):
    slice_condition = np.logical_and(ptr_content["series"] == 0, ptr_content["index"] == 0) #condition to get only one frame for each slice
    slices = ptr_content[slice_condition]["slice"]
    return slices

def calculate_centroid(coords):
    center_x = (max(coords[0]) + min(coords[0])) /2  
    center_y = (max(coords[1]) + min(coords[1])) /2 

    centroid = [center_x, center_y]

    return centroid

def calculate_edge_length(centroid, coords):
    # calculates half the edge length!!!
    edge_length_x = max(coords[0])-centroid[0]
    edge_length_y = max(coords[1])-centroid[1]

    edge_length = max(edge_length_x, edge_length_y) 
    edge_length = edge_length + (edge_length*0.3)

    return edge_length

def translate_coordinates(coordinates, translation):
    new_x_coords = []
    new_y_coords = []
    for i in range(len(coordinates[0])):
        new_x_coords.append(coordinates[0][i]+translation[0])
        new_y_coords.append(coordinates[1][i]+translation[1])

    new_coords = [new_x_coords, new_y_coords]

    return new_coords

'''
This function is used to send an email together with the log file (google server)
Input:  from_addr = email address you want to use to send the email (have to allow low level software - google how to do this)
        to_addr = email address you're sending the email to
        subject = subject of the email
        message = body of the email
        filepath = filepath to the log file
'''
def sendemail(from_addr, to_addr, subject, message, filepath):
    msg = MIMEMultipart()

    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject

    msg.attach(MIMEText(message,"plain"))

    filename = os.path.basename(filepath)
    attachment = open(filepath, "rb")

    part = MIMEBase("application", "octet-stream")
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment; filename = {}".format(filename))

    msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(from_addr, "Justbeaninja13")
    text = msg.as_string()
    server.sendmail(from_addr, to_addr, text)
    server.quit()
    
    return
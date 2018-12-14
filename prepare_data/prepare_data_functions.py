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

def calculate_time_elapsed(start):
    end = time()
    hrs = (end-start)//60//60
    mins = ((end-start) - hrs*60*60)//60
    secs = int((end-start) - mins*60 - hrs*60*60)

    return hrs, mins, secs

def log_and_print(output_messages):
    if isinstance(output_messages, str):
        logger.info(output_messages)
        print(output_messages)
    else:    
        for message in output_messages:
            logger.info(message)
            print(message)

def log_error_and_print(output_messages):
    if isinstance(output_messages, str):
        logger.error(output_messages)
        print(output_messages)
    else:    
        for message in output_messages:
            logger.error(message)
            print(message)

def get_cim_path(patient_name, cim_patients):
    if patient_name != "4J_Y5_B5__XN":  #unique case where this patient has two underscores
        cim_ptr_path = [p for p in cim_patients if patient_name.replace("_Bio", "").lower() in p.lower()][0] #get the cim path of current patient
        cim_pat_name = os.path.basename(cim_ptr_path)
        cim_model_name = os.path.basename(os.path.dirname(cim_ptr_path))
        cim_path = "{}\\{}".format(cim_model_name, cim_pat_name)
    else:    
        cim_path = "CIM_DATA_ze_2\\4J_Y5_B5_XN_ze"

    return cim_path

def get_cim_patients(cim_dir, cim_models):
    # put the paths of the models in a list
    cim_models_paths = [os.path.join(cim_dir, d) for d in os.listdir(cim_dir) if d in cim_models]
    cim_patients = []
    # obtain the cim path of each patient
    for cim_model in cim_models_paths:
        cim_patients += [os.path.join(cim_model, d) for d in os.listdir(cim_model) if os.path.isdir(os.path.join(cim_model, d))]
    
    return cim_patients

def get_slices(ptr_content):
    slice_condition = np.logical_and(ptr_content["series"] == 0, ptr_content["index"] == 0) #condition to get only one frame for each slice
    slices = ptr_content[slice_condition]["slice"]
    return slices

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
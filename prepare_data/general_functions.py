"""
This script mainly contains functions needed for logging
Author: Amos Rada
Date:   22/02/2019
"""
# import needed libraries
import logging
import os
import numpy as np
from time import time
#import libraries to send email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

logger = logging.getLogger(__name__)

def calculate_time_elapsed(start):
    '''
    This function calculates the time elapsed
    Input:  
        start = start time
    Output: 
        hrs, mins, secs = time elapsed in hours, minutes, seconds format
    '''
    end = time()
    hrs = (end-start)//60//60
    mins = ((end-start) - hrs*60*60)//60
    secs = int((end-start) - mins*60 - hrs*60*60)

    return hrs, mins, secs

def log_and_print(output_messages):
    '''
    This function logs info level messages and prints them
    Input:  
        output_messages (string or list of strings) = messages to be added to the log and printed on the terminal
    '''
    if isinstance(output_messages, str):    #if output message is a single string
        logger.info(output_messages)
        print(output_messages)
    else:    
        for message in output_messages:
            logger.info(message)
            print(message)

def log_error_and_print(output_messages):
    '''
    This function logs error level messages and prints them
    Input:  
        output_messages (string or list of strings) = errors to be added to the log and printed on the terminal
    '''
    if isinstance(output_messages, str):    #if error message is a single string
        logger.error(output_messages)
        print(output_messages)
    else:    
        for message in output_messages:
            logger.error(message)
            print(message)

def sendemail(from_addr, to_addr, subject, message, filepath):
    '''
    This function is used to send an email together with the log file (google server)
    Input:  
        from_addr (string) = email address you want to use to send the email (have to allow low level software - google how to do this)
        to_addr (string) = email address you're sending the email to
        subject (string) = subject of the email
        message (string) = body of the email
        filepath (string) = filepath to the log file
    '''
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

# ========== for preparing image pointers ==========
def prompt_user():
    '''
    Prompts user to enter which functions they want to use
    Output:
        steps (list of int) = the number of the function/s that the user wants to use
    '''
    print("Function 1: Create image pointers")
    print("Function 2: Move image pointers with missing data")
    print("Function 3: Find matches between series of interest")
    try:
        steps = [int(s) for s in input("Enter the function(s) you want to use (separated by space): ").split()]
        steps.sort()
        print()
    except ValueError:
        print()
        print("Invalid values entered")
        print("Please enter values from 1 to 3 separate by a space")
        steps = prompt_user()
    
    return steps
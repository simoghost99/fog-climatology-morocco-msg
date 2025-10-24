import datetime
import os
import shutil
import time
import eumdac
import requests
import xarray as xr
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from collections import defaultdict

def download_netcdf(consumer_key, consumer_secret, start, end, step, destination_folder):
    # Check if the destination folder exists, otherwise create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Authenticate with the EUMETSAT Data Access Client API
    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)
    datastore = eumdac.DataStore(token)
    datatailor = eumdac.DataTailor(token)

    # Select the MSG SEVIRI data collection
    selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI')

    # Define the processing chain (data customization)
    chain = eumdac.tailor_models.Chain(
        product='HRSEVIRI',  # Satellite product
        format='netcdf4',  # Output format
        roi={'NSWE': [38, 20, -20, 0]},  # Region of interest (North, South, West, East)
        projection='geographic',  # Geographic projection
        # 1 - VIS0.6, 2 - VIS0.8, 3 - NIR1.6, 4 - IR3.9, 7 - IR8.7, 9 - IR10.8, 10 - IR12.0, 11 - IR13.4
        filter={'bands': ['hrv']}   
    )

    # Generate all time steps
    time_steps = []
    current_time = start
    while current_time < end:
        time_steps.append(current_time)
        current_time = current_time + step
    
    # Calculate total months and days for progress tracking
    total_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    current_month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Dictionaries to track timing information
    month_times = defaultdict(float)
    day_times = defaultdict(float)
    
    failed_customisations = []  # List of failed customisations
    
    # Main progress bar for overall progress
    with tqdm(total=len(time_steps), desc="Overall Progress", unit="file") as overall_pbar:
        current_month_progress = None
        current_day_progress = None
        
        for current_time in time_steps:
            month_key = f"{current_time.year}-{current_time.month:02d}"
            day_key = f"{current_time.year}-{current_time.month:02d}-{current_time.day:02d}"
            
            # Update month progress bar if needed
            if current_month_progress is None or month_key != current_month_progress.desc.split()[-1]:
                if current_month_progress is not None:
                    current_month_progress.close()
                month_steps_in_month = len([t for t in time_steps 
                                          if f"{t.year}-{t.month:02d}" == month_key])
                current_month_progress = tqdm(total=month_steps_in_month, 
                                            desc=f"Month {month_key}", 
                                            unit="file", 
                                            leave=False)
            
            # Update day progress bar if needed
            if current_day_progress is None or day_key != current_day_progress.desc.split()[-1]:
                if current_day_progress is not None:
                    # Record day time before closing
                    day_end_time = time.time()
                    day_times[day_key] = day_end_time - day_start_time
                    current_day_progress.close()
                
                day_steps_in_day = len([t for t in time_steps 
                                      if f"{t.year}-{t.month:02d}-{t.day:02d}" == day_key])
                current_day_progress = tqdm(total=day_steps_in_day, 
                                          desc=f"Day {day_key}", 
                                          unit="file", 
                                          leave=False)
                day_start_time = time.time()  # Start timing the day
            
            step_start_time = time.time()
            
            try:
                dtstart = current_time.replace(minute=0, second=0) - datetime.timedelta(minutes=5)
                dtend = current_time.replace(minute=0, second=0) + datetime.timedelta(minutes=5) 
                
                # Search for available products within the given time range
                products = list(selected_collection.search(dtstart=dtstart, dtend=dtend))

                if not products:
                    tqdm.write(f'[{current_time}] No product found.')
                    # Update progress bars
                    current_month_progress.update(1)
                    current_day_progress.update(1)
                    overall_pbar.update(1)
                    continue

                # Create a customization with the first product found
                customisation = datatailor.new_customisation(products[0], chain)
                tqdm.write(f'[{current_time}] Customisation {customisation._id} started.')

                sleep_time = 5  # Initial waiting time between status checks
                customisation_pbar = tqdm(total=100, desc=f"Customisation {current_time}", leave=False)
                
                while True:
                    status = customisation.status  # Check customisation status
                    
                    # Update customisation progress based on status
                    if "QUEUED" in status:
                        customisation_pbar.set_description(f"Customisation {current_time} - Queued")
                        customisation_pbar.n = 25
                    elif "RUNNING" in status:
                        customisation_pbar.set_description(f"Customisation {current_time} - Running")
                        customisation_pbar.n = 50
                    elif "DONE" in status:
                        customisation_pbar.set_description(f"Customisation {current_time} - Done")
                        customisation_pbar.n = 100
                        customisation_pbar.refresh()
                    
                    tqdm.write(f'[{current_time}] Status: {status}')

                    if "DONE" in status:
                        tqdm.write(f'[{current_time}] Customisation completed successfully.')
                        
                        # Retrieve the generated NetCDF files
                        nc_files = [file for file in customisation.outputs if file.endswith(".nc")]

                        if not nc_files:
                            tqdm.write("No NetCDF file found.")
                        else:
                            for file in nc_files:
                                output_path = os.path.join(destination_folder, os.path.basename(file))
                                with customisation.stream_output(file) as stream, open(output_path, 'wb') as fdst:
                                    shutil.copyfileobj(stream, fdst)  # Download the file
                                tqdm.write(f"Downloaded NetCDF file: {output_path}")
                        break
                    elif status in ["ERROR", "FAILED", "DELETED", "KILLED", "INACTIVE"]:
                        tqdm.write(f'[{current_time}] Error during customisation: {status}')
                        tqdm.write(f'Log: {customisation.logfile}')
                        failed_customisations.append((current_time, customisation._id, status))
                        
                        # If status is "FAILED", delete the old customisation and retry
                        tqdm.write(f'[{current_time}] Retrying customisation...')

                        customisation.delete()
                        tqdm.write(f'[{current_time}] Customisation deleted.')

                        # Create a new customisation to retry
                        customisation = datatailor.new_customisation(products[0], chain)
                        continue

                    elif status == "QUEUED":
                        sleep_time = min(sleep_time + 1, 10)
                    time.sleep(sleep_time)
                    customisation_pbar.refresh()

                customisation_pbar.close()
                
                # Delete the customisation after data retrieval
                customisation.delete()
                tqdm.write(f'[{current_time}] Customisation deleted.')

            except eumdac.datatailor.DataTailorError as error:
                tqdm.write(f'Data Tailor error: {error}')
            except requests.exceptions.RequestException as error:
                tqdm.write(f'Network error: {error}')
            except Exception as error:
                tqdm.write(f'Unexpected error: {error}')
            
            # Calculate and record step time
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            month_times[month_key] += step_time
            
            # Update progress bars
            current_month_progress.update(1)
            current_day_progress.update(1)
            overall_pbar.update(1)
        
        # Close the final day progress bar
        if current_day_progress is not None:
            day_end_time = time.time()
            day_times[day_key] = day_end_time - day_start_time
            current_day_progress.close()
        
        # Close the final month progress bar
        if current_month_progress is not None:
            current_month_progress.close()
    
    # Print timing summary
    print("\n" + "="*50)
    print("DOWNLOAD TIME SUMMARY")
    print("="*50)
    
    print("\nTime per month:")
    for month, total_time in sorted(month_times.items()):
        hours = total_time / 3600
        print(f"  {month}: {total_time:.2f} seconds ({hours:.2f} hours)")
    
    print("\nTime per day:")
    for day, total_time in sorted(day_times.items()):
        hours = total_time / 3600
        print(f"  {day}: {total_time:.2f} seconds ({hours:.2f} hours)")
    
    total_download_time = sum(month_times.values())
    total_hours = total_download_time / 3600
    print(f"\nTotal download time: {total_download_time:.2f} seconds ({total_hours:.2f} hours)")
    
    # Display failed customisations
    if failed_customisations:
        print("\n" + "="*50)
        print("FAILED CUSTOMISATIONS")
        print("="*50)
        for failed in failed_customisations:
            print(f"Date: {failed[0]} | ID: {failed[1]} | Status: {failed[2]}")
    else:
        print("\nNo failed customisations!")

# Authentication parameters
CREDENTIALS_FILE = os.path.join(os.path.expanduser("~"), '.eumetsat_api_key')

with open(CREDENTIALS_FILE) as f:
    creds = json.load(f)

consumer_key = creds['consumer_key']
consumer_secret = creds['consumer_secret']

# Define the download period
start = datetime.datetime(2024, 1, 1, 0, 0)
end = datetime.datetime(2024, 6, 30, 23, 0)
step = datetime.timedelta(minutes=60)  # Download interval (1 hour)

destination_folder = 'data/image_sat'

download_netcdf(consumer_key, consumer_secret, start, end, step, destination_folder)

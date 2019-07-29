
import logging
import csv
import os
from my_utils import makedir_and_clear

def write_discharge_summaries(MIMIC_3_DIR, output_dir):
    notes_file = '%s/NOTEEVENTS.csv' % (MIMIC_3_DIR)

    makedir_and_clear(output_dir)

    logging.info("processing notes file")

    count_note = 0

    with open(notes_file, 'r') as csvfile:

        notereader = csv.reader(csvfile)
        #header
        next(notereader)

        for line in notereader: # each line is saved as a file
            subj = int(line[1])
            category = line[6]
            if category == "Discharge summary":
                file_name = line[1]+'_'+line[2]+".txt"
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, 'w') as outfile:
                    logging.info("writing to %s" % (file_path))
                    outfile.write(line[10])
                count_note += 1

    logging.info("totally write {} notes".format(count_note))
    return


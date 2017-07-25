
import csv, logging


def csv2listofdicts(filename_csv):
    logging.info('Reading list of dicts from: ' + filename_csv)
    listofdicts = []
    with open(filename_csv, 'r') as fid:
        csv.list_dialects()
        reader = csv.DictReader(fid, delimiter=',')
        for row in reader: 
            tmp = {}
            for fieldname in reader.fieldnames:
                tmp[fieldname] = eval(row[fieldname])
            listofdicts.append(tmp)
    logging.info('Generated list of {} dicts with the following field names: {}'.format(len(listofdicts), reader.fieldnames)) 
    return listofdicts
            
            
def repr2listofdicts(filename):
    from numpy import array
    with open(filename, 'r') as fid:
        trimcase_str = fid.read()
    trimcase = eval(trimcase_str)
    logging.info('Generated list of {} dicts from: {}'.format(len(trimcase), filename) )
    return trimcase

def generate_empty_listofdicts(trimcase):
    empty_listofdicts = [{} for i in trimcase]
    logging.info('Generated list of {} empty dicts.'.format(len(empty_listofdicts)) )
    return empty_listofdicts
import copy, os, logging
import pandas as pd

from loadskernel.io_functions.bdf_cards import *

"""
This is a simple and light-weight BDF reader which consist of only two scripts (read_bdf.py & bdf_cards.py),
and parses Nastran BDF files to Pandas data frames. It considers only those cards and fields actually needed
for a loads analysis using Loads Kernel, providing maximum compatibility and speed and comes without further 
dependencies.
Some ideas and concepts are inspired by pyBDF, a comprehensive DLR in-house BDF reader by Markus Zimmer.
"""

class Reader(object):
    # This is the list (and mapping) of all implemented bdf cards.
    card_interpreters = {'GRID'   : GRID,
                         'CQUAD4' : CQUAD4,
                         'CTRIA3' : CTRIA3,
                         'CORD2R' : CORD2R,
                         'CORD1R' : CORD1R,
                         'MONPNT1': MONPNT1,
                         'AECOMP' : AECOMP,
                         'SET1'   : SET1,
                         'AEFACT' : AEFACT,
                         'CAERO1' : CAERO1,
                         'CAERO7' : CAERO7,
                         'AESURF' : AESURF,
                         'AELIST' : AELIST,
                         }

    def __init__(self):
        # This is the storage for processed files
        self.processed_files = []
        # This is the line storage
        self.lines = []
        # This is the include storage
        self.includes = []
        # This is a list of all known / implemented cards
        self.known_cards = self.card_interpreters.keys()
        # The cards are stored in a Pandas data frames, one frame per type of card. 
        # The data frames themselfes are stored in a dictionary. 
        self.cards = {}
        for card_name in self.known_cards:
            # Make sure the colums of the data frame have the correct type (int or float)
            # See https://stackoverflow.com/questions/36462257/create-empty-dataframe-in-pandas-specifying-column-types
            # However, the dtype changes when adding a new row. Not sure if Pandas is the best tool for this job...
            card_class = self.card_interpreters[card_name]
            df_definition = {}
            for c, t in zip(card_class.field_names, card_class.field_types):
                df_definition[c] = pd.Series(dtype=t)
            # Create the data frame 
            self.cards[card_name] = pd.DataFrame(df_definition)
        
    def process_deck(self, deck):
        # Make sure deck is a list of filenames, not a single string
        if isinstance(deck, str):
            self.filenames = [deck]
        else:
            self.filenames = deck
        """
        This is the iterative loop to capture all include statements.
        Step 1: In case include statements are found, move them to filenames.
        Step 2: Re-run process_deck()
        Step 3: This loop terminates when the include list is empty, i.e. no more includes are found.
        """ 
        self.read_lines_from_files()
        self.read_cards_from_lines()
        
        if self.includes:
            logging.info('Found include(s):')
            self.filenames = copy.deepcopy(self.includes)
            self.includes = []
            self.process_deck()
        
        self.remove_duplicate_cards()
        return
    
    def read_lines_from_files(self):
        # reset the line storage before reading new files 
        self.lines = []
        # loop over all filenames and read all lines
        for filename in self.filenames:
            # to save time, make sure the same file is not parsed twice
            if filename in self.processed_files:
                logging.info('File already processed: {}'.format(filename))
            # make sure the given filename exists, if not, skip that file
            elif os.path.exists(filename):
                logging.info('Read from file: {}'.format(filename))
                with open(filename, 'r') as fid:
                    self.lines += fid.readlines()
                self.processed_files += [filename]
            else:
                logging.warning('File NOT found: {}'.format(filename))
    
    def read_cards_from_lines(self):
        if self.lines:
            logging.info('Read BDF cards from {} lines...'.format(len(self.lines)))
        # loop over all lines until empty
        while self.lines:
            # test the first 8 characters of the line for a known card
            card_name = self.lines[0][:8].strip('*, ').upper()
            if card_name in self.known_cards:
                # get the corresponding interpeter
                card_class = self.card_interpreters[card_name]
                # convert lines to string              
                lines_as_string, width = self.convert_lines_to_string(card_class.expected_lines)
                # parse that string using the proper interpreter
                card = card_class.parse(lines_as_string, width)
                # store the card
                self.store_card(card_name, card)
            else:      
                self.lines.pop(0)
            
    def convert_lines_to_string(self, expected_lines):
        width = self.get_width_of_fields(self.lines)
        if expected_lines is not None:
            # This is the simple case where the numer of lines is already known.
            n_lines = expected_lines
        else:
            """
            This is the more complex case where the number of lines in unknown.
            Strategy: Loop over the lines and see if the continuation character appears again in the next line.
            This should work with any continuation character and also for implicit line continuation (where no
            continuation character is given).
            """
            for i, line in enumerate(self.lines):
                if len(self.lines) == i+1 or line[9*width:].strip() != self.lines[i+1][:width].strip(): 
                    n_lines = i+1
                    break
        # get the line to work with
        my_lines = self.lines[:n_lines]
        # remove line breakes (Linux and Windows) at the end of the lines
        my_lines = [line.strip('\n') for line in my_lines]
        my_lines = [line.strip('\r') for line in my_lines]
        # Trim the lines:
        # - remove trailing / continuation characters
        # - remove first field, this is either the card name (which we no longer need) or the continuation character
        # - expand a line which is missing spaces at the end, which is important for indexing the fields
        # - handle that the last line or a one-line-card might have less than 9 fields  
        if n_lines > 1:
            tmp = []
            for line in my_lines[:-1]:
                n_missing_spaces = (8*width - len(line[width:9*width]))
                tmp.append(line[width:9*width]+' '*n_missing_spaces)
            tmp.append(my_lines[-1][width:])
            my_lines = tmp
        else:
            my_lines = [my_lines[-1][width:]]
        
        # Join lines to one string
        lines_as_string = ''.join(my_lines)
        # Removing consumed lines from list, pop only accepts one index
        for _ in range(n_lines):
            self.lines.pop(0)
        return lines_as_string, width
        
    def get_width_of_fields(self, lines):
        # Establish the width of the fields in the Nastran card, which can be 8 or 16 characters.
        # This is indicated by a '*' at the beginning of a the card.
        if lines[0][0] == '*':
            width = 16
        else:
            width = 8
        return width
    
    def store_card(self, card_name, card):
        # ToDo: It would be nice if the dtype would be conserved.
        new_row = pd.Series(card)
        self.cards[card_name] = pd.concat([self.cards[card_name], new_row.to_frame().T], ignore_index=True)
    
    def remove_duplicate_cards(self):
        # This function looks for duplicates in all data frames.
        # Duplicates are identified by the first field (typically ID or NAME).
        for card_name in self.known_cards:
            old_size = self.cards[card_name].shape[0]
            sort_by_field = self.card_interpreters[card_name].field_names[0]
            self.cards[card_name].drop_duplicates(sort_by_field, inplace=True)
            new_size = self.cards[card_name].shape[0]
            if old_size != new_size:
                logging.info('Dropping {} duplicate {}s'.format(old_size-new_size, card_name))


# bdf_reader = Reader()
# bdf_reader.process_deck('./allegra-s_c05.bdf')
# bdf_reader.process_deck('./test.bdf')
# print('Done.')

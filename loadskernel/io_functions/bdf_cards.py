import logging
from loadskernel.io_functions.read_mona import nastran_number_converter

class SimpleCard(object):
    
    @classmethod
    def parse(cls, card_as_string, width):
        # create an empty dictionary to store the card
        card = {}
        # loop over all fields
        for name, pos, type in zip(cls.field_names, cls.field_positions, cls.field_types):
            # look for a default value
            if name in cls.optional_fields:
                default_value = cls.optional_defaults[cls.optional_fields.index(name)]
            else:
                default_value = None
            # get the field location in the string
            start = pos*width
            stop = (pos+1)*width
            """
            This is the decision logic to parse fields:
            Case 1: See if the card is suffiently long --> the field exists
            Note that the stop index also works in case the card is too short, it simply take what's there --> this might be the last field
            Case 2: The field doesn't exists --> look if the field is optional and take the default
            Case 3: The field is missing --> issue a warning
            """
            if len(card_as_string) > start:
                my_field = card_as_string[start:stop]
                # convert field and store in dictionary
                card[name] = nastran_number_converter(my_field, type, default_value)
            elif name in cls.optional_fields:
                # store default in dictionary
                card[name] = default_value
            else:
                logging.error('Field {} expected but missing in {} card: {}'.format(name, cls.__name__, card_as_string))

        print(card)
        return card
    
class GRID(SimpleCard):
    expected_lines = 1
    # field of interest (any other fields are not implemented)
    field_names         = ['ID', 'CP', 'X1',   'X2',   'X3',   'CD' ]
    field_positions     = [  0,    1,    2,      3,      4,      5  ]
    field_types         = ['int','int','float','float','float','int']
    optional_fields     = ['CP', 'CD']
    optional_defaults   = [  0,    0 ]

class CQUAD4(SimpleCard):
    expected_lines = 1
    # field of interest (any other fields are not implemented)
    field_names         = ['ID', 'G1', 'G2', 'G3', 'G4' ]
    field_positions     = [  0,    2,    3,    4,    5  ]
    field_types         = ['int','int','int','int','int']
    optional_fields     = []
    optional_defaults   = []

class CTRIA3(SimpleCard):
    expected_lines = 1
    # field of interest (any other fields are not implemented)
    field_names         = ['ID', 'G1', 'G2', 'G3' ]
    field_positions     = [  0,    2,    3,    4  ]
    field_types         = ['int','int','int','int']
    optional_fields     = []
    optional_defaults   = []

class CORD2R(SimpleCard):
    expected_lines = 2
    # field of interest (any other fields are not implemented)
    field_names         = ['ID', 'RID', 'A1',  'A2',   'A3',   'B1',   'B2',   'B3',   'C1',   'C2',   'C3'   ]
    field_positions     = [  0,     1,    2,     3,      4,      5,      6,      7,      8,      9,     10    ]
    field_types         = ['int','int','float','float','float','float','float','float','float','float','float']
    optional_fields     = ['RID']
    optional_defaults   = [   0 ]

class CORD1R(SimpleCard):
    expected_lines = 2
    # field of interest (any other fields are not implemented)
    field_names         = ['ID', 'RID', 'A',  'B',   'C']
    field_positions     = [  0,     1,   2,    3,     4 ]
    field_types         = ['int','int','int','int','int']
    optional_fields     = ['RID']
    optional_defaults   = [   0 ]

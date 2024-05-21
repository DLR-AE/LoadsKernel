import logging

from loadskernel.io_functions.read_mona import nastran_number_converter


class SimpleCard():
    # init empty attributes
    expected_lines = 1
    field_names = []
    field_positions = []
    field_types = []
    optional_fields = []
    optional_defaults = []

    @classmethod
    def parse(cls, card_as_string, width):
        card = cls.parse_known_fields(cls, card_as_string, width)
        return card

    def parse_known_fields(self, card_as_string, width):
        # create an empty dictionary to store the card
        card = {}
        # loop over all fields
        for name, pos, field_type in zip(self.field_names, self.field_positions, self.field_types):
            # look for a default value
            if name in self.optional_fields:
                default_value = self.optional_defaults[self.optional_fields.index(name)]
            else:
                default_value = None
            # get the field location in the string
            start = pos * width
            stop = (pos + 1) * width
            """
            This is the decision logic to parse fields:
            Case 1: See if the card is suffiently long --> the field exists
            Note that the stop index also works in case the card is too short, it simply take what's there --> this might be
            the last field
            Case 2: The field doesn't exists --> look if the field is optional and take the default
            Case 3: The field is missing --> issue a warning
            """
            if len(card_as_string) > start:
                my_field = card_as_string[start:stop]
                # convert field and store in dictionary
                card[name] = nastran_number_converter(my_field, field_type, default_value)
            elif name in self.optional_fields:
                # store default in dictionary
                card[name] = default_value
            else:
                logging.error('Field {} expected but missing in {} card: {}'.format(name, type(self).__name__, card_as_string))
        return card


class ListCard(SimpleCard):

    @classmethod
    def parse(cls, card_as_string, width):
        """
        For the first fields, re-use the procedure from SimpleCard.
        Then, parse further occurences for the last field, which yield the list items.
        """
        card = cls.parse_known_fields(cls, card_as_string, width)
        card = cls.parse_list_items(cls, card, card_as_string, width)
        return card

    def parse_list_items(self, card, card_as_string, width):
        # get the properties of the last field
        name = self.field_names[-1]
        pos = self.field_positions[-1]
        field_type = self.field_types[-1]
        # look for a default value
        if name in self.optional_fields:
            default_value = self.optional_defaults[self.optional_fields.index(name)]
        else:
            default_value = None
        # turn the last field into a list
        card[name] = [card[name]]
        """
        This is the loop to find more occurences for the last field:
        Case 1: See if the card is suffiently long --> append the field to list
        Case 2: The field doesn't exists --> break the loop
        """
        i = 1
        while True:
            # get the field location in the string
            start = (pos + i) * width
            stop = (pos + i + 1) * width

            if len(card_as_string) > start:
                my_field = card_as_string[start:stop]
                # convert field and store in dictionary
                card[name].append(nastran_number_converter(my_field, field_type, default_value))
                i += 1
            else:
                break
        return card


class StringCard(SimpleCard):

    @classmethod
    def parse(cls, card_as_string, _):
        card = cls.parse_string(cls, card_as_string)
        return card

    def parse_string(self, card_as_string):
        # create an empty dictionary to store the card
        card = {}
        name = self.field_names[0]
        """
        This is the decision logic to parse fields:
        Case 1: See if the card is suffiently long --> the field exists
        Case 2: The field is missing --> issue a warning
        """
        if len(card_as_string) > 0:
            # convert field and store in dictionary
            tmp = card_as_string.strip("'*, ")
            card[name] = tmp.replace(" ", "")
        else:
            logging.error('Field {} expected but missing in {} card: {}'.format(name, type(self).__name__, card_as_string))
        return card


class GRID(SimpleCard):
    expected_lines = 1
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'CP', 'X1', 'X2', 'X3', 'CD']
    field_positions = [0, 1, 2, 3, 4, 5]
    field_types = ['int', 'int', 'float', 'float', 'float', 'int']
    optional_fields = ['CP', 'CD']
    optional_defaults = [0, 0]


class CQUAD4(SimpleCard):
    expected_lines = 1
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'G1', 'G2', 'G3', 'G4']
    field_positions = [0, 2, 3, 4, 5]
    field_types = ['int', 'int', 'int', 'int', 'int']
    optional_fields = []
    optional_defaults = []


class CTRIA3(SimpleCard):
    expected_lines = 1
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'G1', 'G2', 'G3']
    field_positions = [0, 2, 3, 4]
    field_types = ['int', 'int', 'int', 'int']
    optional_fields = []
    optional_defaults = []


class CORD2R(SimpleCard):
    expected_lines = 2
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'RID', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
    field_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    field_types = ['int', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']
    optional_fields = ['RID']
    optional_defaults = [0]


class CORD1R(SimpleCard):
    expected_lines = 1
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'A', 'B', 'C']
    field_positions = [0, 1, 2, 3]
    field_types = ['int', 'int', 'int', 'int']
    optional_fields = []
    optional_defaults = []


class MONPNT1(SimpleCard):
    expected_lines = 2
    # field of interest (any other fields are not implemented)
    field_names = ['NAME', 'COMP', 'CP', 'X', 'Y', 'Z', 'CD']
    field_positions = [0, 9, 10, 11, 12, 13, 14]
    field_types = ['str', 'str', 'int', 'float', 'float', 'float', 'int']
    optional_fields = ['CP', 'CD']
    optional_defaults = [0, 0]


class AECOMP(ListCard):
    expected_lines = None
    # field of interest (any other fields are not implemented)
    field_names = ['NAME', 'LISTTYPE', 'LISTID']
    field_positions = [0, 1, 2]
    field_types = ['str', 'str', 'int']
    optional_fields = ['LISTID']
    optional_defaults = [None]


class SET1(ListCard):
    expected_lines = None
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'values']
    field_positions = [0, 1]
    # Due to the mixture of integers and strings ('THRU') in a SET1 card, all list items are parsed as strings.
    field_types = ['int', 'str']
    # Blank strings (e.g. trailing spaces) shall be replaced with None.
    optional_fields = ['values']
    optional_defaults = [None]


class AELIST(SET1):
    # The AELIST is identical to SET1.
    pass


class AEFACT(ListCard):
    expected_lines = None
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'values']
    field_positions = [0, 1]
    field_types = ['int', 'float']
    optional_fields = ['values']
    optional_defaults = [None]


class CAERO1(SimpleCard):
    expected_lines = 2
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'CP', 'NSPAN', 'NCHORD', 'LSPAN', 'LCHORD', 'X1', 'Y1', 'Z1', 'X12', 'X4', 'Y4', 'Z4', 'X43']
    field_positions = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    field_types = ['int', 'int', 'int', 'int', 'int', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                   'float']
    optional_fields = ['CP', 'NSPAN', 'NCHORD', 'LSPAN', 'LCHORD']
    optional_defaults = [0, 0, 0, None, None]


class CAERO7(SimpleCard):
    expected_lines = 3
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'CP', 'NSPAN', 'NCHORD', 'X1', 'Y1', 'Z1', 'X12', 'X4', 'Y4', 'Z4', 'X43']
    field_positions = [0, 2, 3, 4, 8, 9, 10, 11, 16, 17, 18, 19]
    field_types = ['int', 'int', 'int', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']
    optional_fields = ['CP', 'NSPAN', 'NCHORD']
    optional_defaults = [0, 0, 0]


class AESURF(SimpleCard):
    expected_lines = 1
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'LABEL', 'CID', 'AELIST', 'EFF']
    field_positions = [0, 1, 2, 3, 7]
    field_types = ['int', 'str', 'int', 'int', 'float']
    optional_fields = ['EFF']
    optional_defaults = [1.0]


class ASET1(ListCard):
    expected_lines = None
    # field of interest (any other fields are not implemented)
    field_names = ['ID', 'values']
    field_positions = [0, 1]
    # Due to the mixture of integers and strings ('THRU') in a SET1 card, all list items are parsed as strings.
    field_types = ['int', 'str']
    # Blank strings (e.g. trailing spaces) shall be replaced with None.
    optional_fields = ['ID', 'values']
    optional_defaults = [123456, None]


class INCLUDE(StringCard):
    expected_lines = None
    # field of interest (any other fields are not implemented)
    field_names = ['filename']
    field_types = ['str']

"""
This OP2 reader is adopted from pyNastran, which is licensed under the
following conditions. See also https://github.com/SteveDoyle2/pyNastran.


Copyright (c) 2011-2022 Steven Doyle & pyNastran developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the pyNastran developers nor the names of any
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import struct
import numpy as np

#  Notes on the op2 format.
#
#  DATA BLOCK:
#      All data blocks (including header) start with header 3 elements:
#      [reclen, key, endrec]
#        - reclen = 1 32-bit integer that specifies number of bytes in
#          key (either 4 or 8)
#        - key = 4 or 8 byte integer specifying number of words in next
#          record
#        - endrec = reclen
#
#      DATA SET, can be multiple records:
#          Next is [reclen, data, endrec]
#            - reclen = 1 32-bit integer that specifies number of bytes
#              in data
#            - data = reclen bytes long, variable format; may be part of
#              a data set or the complete set
#            - endrec = reclen
#
#          Next is info about whether we're done with current data set:
#          [reclen, key, endrec]
#            - reclen = 1 32-bit integer that specifies number of bytes
#              in key (either 4 or 8)
#            - key = 4 or 8 byte integer specifying number of words in
#              next record; if 0, done with data set
#            - endrec = reclen
#
#          If not done, we have [reclen, data, endrec] for part 2 (and
#          so on) for the record.
#
#      Once data set is complete, we have: [reclen, key, endrec]
#        - reclen = 1 32-bit integer that specifies number of bytes in
#          key (either 4 or 8)
#        - key = 4 or 8 byte integer specifying number of words in next
#          record (I think ... not useful?)
#        - endrec = reclen
#
#      Then: [reclen, rec_type, endrec]
#        - reclen = 1 32-bit integer that specifies number of bytes in
#          rec_type (either 4 or 8)
#        - rec_type = 0 if table (4 or 8 bytes)
#        - endrec = reclen
#
#      Then, info on whether we're done with data block:
#      [reclen, key, endrec]
#        - reclen = 1 32-bit integer that specifies number of bytes in
#          key (either 4 or 8)
#        - key = 4 or 8 byte integer specifying number of words in next
#          record; if 0, done with data block
#        - endrec = reclen
#
#      If not done, we have [reclen, data, endrec] for record 2 and so
#      on, until data block is read in.


class OP2():
    """Class for reading Nastran op2 files and nas2cam data files."""

    def __init__(self, filename=None):
        self._fileh = None
        if isinstance(filename, str):
            self._op2_open(filename)

    def __del__(self):
        if self._fileh:
            self._fileh.close()
            self._fileh = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self._fileh:
            self._fileh.close()
            self._fileh = None
        return False

    def _op2_open(self, filename):
        """
        Open op2 file in correct endian mode.

        Sets these class variables:

        _fileh : file handle
            Value returned by open().
        _swap : bool
            True if bytes must be swapped to correct endianness.
        _bit64 : True or False
            True if 'key' integers are 64-bit.
        _endian : string
            Will be '=' if `swap` is False; otherwise, either '>' or '<'
            for big-endian and little-endian, respectively.
        _intstr : string
            Either `endian` + 'i4' or `endian` + 'i8'.
        _ibytes : integer
            Either 4 or 8 (corresponds to `intstr`)
        _int32str : string
           `endian` + 'i4'.
        _label : string
            The op2 header label or, if none, None.
        _date : vector
            Three element date vector, or None.
        _nastheader : string
            Nastran header for file, or None.
        _postheaderpos : integer
            File position after header.
        dbnames : dictionary
            See :func:`directory` for description.  Contains data block
            names, bytes in file, file positions, and for matrices, the
            matrix size.
        dblist : list
            See :func:`directory` for description.  Contains same info
            as dbnames, but in a list of ordered and formatted strings.
        _Str4 : struct.Struct object
            Precompiled for reading 4 byte integers (corresponds to
            `int32str`).
        _Str : struct.Struct object
            Precompiled for reading 4 or 8 byte integers (corresponds
            to `intstr`).

        File is positioned after the header label (at `postheaderpos`).
        """
        self._fileh = open(filename, 'rb')
        self.dbnames = []
        self.dblist = []
        reclen = struct.unpack('i', self._fileh.read(4))[0]
        self._fileh.seek(0)

        reclen = np.array(reclen, dtype=np.int32)
        if not np.any(reclen == [4, 8]):
            self._swap = True
            reclen = reclen.byteswap()
            if not np.any(reclen == [4, 8]):
                self._fileh.close()
                self._fileh = None
                raise RuntimeError('Could not decipher file.  First'
                                   '4-byte integer should be 4 or 8.')
            if sys.byteorder == 'little':
                self._endian = '>'
            else:
                self._endian = '<'
        else:
            self._swap = False
            self._endian = '='

        self._Str4 = struct.Struct(self._endian + 'i')
        if reclen == 4:
            self._bit64 = False
            self._intstr = self._endian + 'i4'
            self._intstru = self._endian + '%di'
            self._ibytes = 4
            self._Str = self._Str4
        else:
            self._bit64 = True
            self._intstr = self._endian + 'i8'
            self._intstru = self._endian + '%dq'
            self._ibytes = 8
            self._Str = struct.Struct(self._endian + 'q')
        # print('bit64 = ', self._bit64)

        self._rowsCutoff = 3000
        self._int32str = self._endian + 'i4'
        self._int32stru = self._endian + '%di'
        self._read_op2_header()
        self._postheaderpos = self._fileh.tell()
        self.directory(verbose=True)

    def _get_key(self):
        """Reads [reclen, key, endrec] triplet and returns key."""
        self._fileh.read(4)
        key = self._Str.unpack(self._fileh.read(self._ibytes))[0]
        self._fileh.read(4)
        return key

    def _skip_key(self, n):
        """Skips `n` key triplets ([reclen, key, endrec])."""
        self._fileh.read(n * (8 + self._ibytes))

    def _read_op2_header(self):
        """
        Returns Nastran output2 header label (or 'no header').
        """
        key = self._get_key()
        if key != 3:
            self._fileh.seek(0)
            self._date = self._nastheader = self._label = None
            return

        self._fileh.read(4)  # reclen
        frm = self._intstru % key
        bytes = self._ibytes * key
        self._date = struct.unpack(frm, self._fileh.read(bytes))
        # self._date = np.fromfile(self._fileh, self._intstr, key)
        self._fileh.read(4)  # endrec
        self._get_key()

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        self._nastheader = self._fileh.read(reclen).decode()
        self._fileh.read(4)  # endrec
        self._get_key()

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        self._label = self._fileh.read(reclen).decode().\
            strip().replace(' ', '')
        self._fileh.read(4)  # endrec
        self._skip_key(2)

    def _read_op2_end_of_table(self):
        """Read Nastran output2 end-of-table marker.

        Returns
        -------
        tuple: (eot, key)
            eot : integer
                1 if end-of-file has been reached and 0 otherwise.
            key : integer
                0 of eot is 1; next key value otherwise.
        """
        bstr = self._fileh.read(4)  # reclen
        if len(bstr) == 4:
            key = self._Str.unpack(self._fileh.read(self._ibytes))[0]
            self._fileh.read(4)  # endrec
        else:
            key = 0
        if key == 0:
            return 1, 0
        return 0, key

    def _read_op2_name_trailer(self):
        """Read Nastran output2 datablock name and trailer.

        Returns
        -------
        tuple: (name, trailer, type)
            name : string
                Name of upcoming data block (upper case).
            trailer : tuple
                Data block trailer.
            type : 0 or 1
                0 means table, 1 means matrix.  I think.

        All outputs will be None for end-of-file.
        """
        eot, key = self._read_op2_end_of_table()
        if key == 0:
            # print('return None, None, None')
            return None, None, None

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        db_binary_name = self._fileh.read(reclen)
        db_name = db_binary_name.strip().decode('ascii')
        self._fileh.read(4)  # endrec
        self._get_key()
        key = self._get_key()

        self._fileh.read(4)  # reclen
        frm = self._intstru % key
        nbytes = self._ibytes * key

        # prevents a giant read
        assert nbytes > 0, nbytes
        trailer = struct.unpack(frm, self._fileh.read(nbytes))
        # trailer = np.fromfile(self._fileh, self._intstr, key)
        self._fileh.read(4)  # endrec
        self._skip_key(4)

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        self._fileh.read(reclen)
        self._fileh.read(4)  # endrec

        self._skip_key(2)
        rec_type = self._get_key()
        return db_name, trailer, rec_type

    def skip_op2_matrix(self, trailer):
        """
        Skip Nastran op2 matrix at current position.

        It is assumed that the name has already been read in via
        :func:`_read_op2_name_trailer`.

        The size of the matrix is read from trailer:
             rows = trailer[2]
             cols = trailer[1]
        """
        dtype = 1
        while dtype > 0:  # read in matrix columns
            # key is number of elements in next record (row # followed
            # by key-1 real numbers)
            key = self._get_key()
            # skip column
            while key > 0:
                reclen = self._Str4.unpack(self._fileh.read(4))[0]
                self._fileh.seek(reclen, 1)
                self._fileh.read(4)  # endrec
                key = self._get_key()
            self._get_key()
            dtype = self._get_key()
        self._read_op2_end_of_table()

    def skip_op2_table(self):
        """Skip over Nastran output2 table."""
        _, key = self._read_op2_end_of_table()
        if key == 0:
            return
        while key > 0:
            while key > 0:
                reclen = self._Str4.unpack(self._fileh.read(4))[0]
                self._fileh.seek(8 + reclen, 1)
                key = self._Str.unpack(self._fileh.read(self._ibytes))[0]
                self._fileh.read(4)  # endrec
            self._skip_key(2)
            eot, key = self._read_op2_end_of_table()

    def print_data_block_directory(self):
        """
        Prints op2 data block directory.  See also :func:`directory`.
        """
        if len(self.dblist) == 0:
            self.directory(verbose=False)
        for s in self.dblist:
            print(s)

    def directory(self, verbose=True, redo=False):  # TODO: _read_op2_name_trailer
        """
        Return list of data block names in op2 file.

        Parameters
        ----------
        verbose : bool (or any true/false variable)
            If True, print names, sizes, and file offsets to screen.
        redo : bool
            If True, scan through file and redefine self.dbnames even
            if it is already set.

        Returns tuple: (dbnames, dblist)
        --------------------------------
        dbnames : Dictionary
            Dictionary indexed by data block name.  Each value is a
            list, one element per occurrence of the data block in the
            op2 file.  Each element is another list that has 3
            elements: [fpos, bytes, size]:
            ::
               fpos : 2-element list; file position start and stop
                      (stop value is start of next data block)
               bytes: number of bytes data block consumes in file
               size : 2-element list; for matrices, [rows, cols],
                      for tables [0, 0]
        dblist : list
            List of strings for printing.  Contains the info above
            in formatted and sorted (in file position order) strings.

        As an example of using dbnames, to get a list of all sizes of
        matrices named 'KAA':
        ::
            o2 = op2.OP2('mds.op2')
            s = [item[2] for item in o2.dbnames['KAA']]

        For another example, to read in first matrix named 'KAA':
        ::
            o2 = op2.OP2('mds.op2')
            fpos = o2.dbnames['KAA'][0][0][0]
            o2._fileh.seek(fpos)
            name, trailer, rectype = o2._read_op2_name_trailer()
            kaa = o2.read_op2_matrix(trailer)

        This routine also sets self.dbnames = dbnames.
        """
        if len(self.dbnames) > 0 and not redo:
            return self.dbnames
        dbnames = {}
        dblist = []
        self._fileh.seek(self._postheaderpos)
        pos = self._postheaderpos
        while 1:
            name, trailer, dbtype = self._read_op2_name_trailer()
            if name is None:
                break
            if dbtype > 0:
                self.skip_op2_matrix(trailer)
                size = [trailer[2], trailer[1]]
                s = 'Matrix {0:8}'.format(name)
            else:
                self.skip_op2_table()
                size = [0, 0]
                s = 'Table  {0:8}'.format(name)
            cur = self._fileh.tell()
            s += (', bytes = {0:10} [{1:10} to {2:10}]'.
                  format(cur - pos - 1, pos, cur))
            if size != [0, 0]:
                s += (', {0:6} x {1:<}'.
                      format(size[0], size[1]))
            if name not in dbnames:
                dbnames[name] = []
            dbnames[name].append([[pos, cur], cur - pos - 1, size])
            dblist.append(s)
            pos = cur
        self.dbnames = dbnames
        self.dblist = dblist
        if verbose:
            self.print_data_block_directory()
        return dbnames, dblist

    def read_op2_record(self, form=None, N=0):
        """
        Read Nastran output2 data record.

        Parameters
        ----------
        form : string or None
            String specifying format, or None to read in signed integers.
            One of::
               'int' (same as None)
               'uint'
               'single'
               'double'
               'bytes' -- raw bytes from file
        N : integer
            Number of elements in final data record; use 0 if unknown.

        Returns numpy 1-d vector or, if form=='bytes', a bytes string.

        This routine will read in a 'super' record if the data spans
        more than one logical record.
        """
        key = self._get_key()
        f = self._fileh
        if not form or form == 'int':
            frm = self._intstr
            frmu = self._intstru
            bytes_per = self._ibytes
        elif form == 'uint':
            frm = self._intstr.replace('i', 'u')
            frmu = self._intstru.replace('i', 'I')
            bytes_per = self._ibytes
        elif form == 'double':
            frm = self._endian + 'f8'
            frmu = self._endian + '%dd'
            bytes_per = 8
        elif form == 'single':
            frm = self._endian + 'f4'
            frmu = self._endian + '%df'
            bytes_per = 4
        elif form == 'bytes':
            data = b''
            while key > 0:
                reclen = self._Str4.unpack(f.read(4))[0]
                data += f.read(reclen)
                f.read(4)  # endrec
                key = self._get_key()
            self._skip_key(2)
            return data
        else:
            raise ValueError("form must be one of:  None, 'int', "
                             "'uint', 'double', 'single' or 'bytes'")
        if N:
            # print('frm=%r' % frm)
            data = np.zeros(N, dtype=frm)
            i = 0
            while key > 0:
                reclen = self._Str4.unpack(f.read(4))[0]
                # f.read(4)  # reclen
                n = reclen // bytes_per
                if n < self._rowsCutoff:
                    b = n * bytes_per
                    # print('frmu=%r' % frmu)
                    data[i:i + n] = struct.unpack(frmu % n, f.read(b))
                else:
                    data[i:i + n] = np.fromfile(f, frm, n)
                i += n
                f.read(4)  # endrec
                key = self._get_key()
        else:
            data = np.zeros(0, dtype=frm)
            while key > 0:
                reclen = self._Str4.unpack(f.read(4))[0]
                # f.read(4)  # reclen
                n = reclen // bytes_per
                if n < self._rowsCutoff:
                    b = n * bytes_per
                    cur = struct.unpack(frmu % n, f.read(b))
                else:
                    cur = np.fromfile(f, frm, n)
                data = np.hstack((data, cur))
                f.read(4)  # endrec
                key = self._get_key()
        self._skip_key(2)
        return data

    def skip_op2_record(self):
        """
        Skip over Nastran output2 data record (or super-record).
        """
        key = self._get_key()
        while key > 0:
            reclen = self._Str4.unpack(self._fileh.read(4))[0]
            self._fileh.seek(reclen + 4, 1)
            key = self._get_key()
        self._skip_key(2)

    def _read_op2_uset(self):
        """
        Read the USET data block.

        Returns 1-d USET array.  The 2nd bit is cleared for the S-set.

        See :func:`rdn2cop2`.
        """
        uset = self.read_op2_record('uint')
        # clear the 2nd bit for all S-set:
        s = 1024 | 512
        sset = 0 != (uset & s)
        if any(sset):
            uset[sset] = uset[sset] & ~2
        self._read_op2_end_of_table()
        return uset


def read_post_op2(op2_filename, verbose=False):
    """
    Reads PARAM,POST,-1 op2 file and returns dictionary of data.

    Parameters
    ----------
    op2_filename : string
        Name of op2 file.
    verbose : bool
        If true, echo names of tables and matrices to screen

    Returns dictionary with following members
    -----------------------------------------
    'uset' : array
    """
    # read op2 file:
    with OP2(op2_filename) as o2:
        uset = None
        o2._fileh.seek(o2._postheaderpos)

        while 1:
            name, trailer, dbtype = o2._read_op2_name_trailer()
            # print('name = %r' % name)
            # print('trailer = %s' % str(trailer))
            # print('dbtype = %r' % dbtype)
            if name is None:
                break
            if name == '':
                raise RuntimeError('name=%r' % name)
            if dbtype > 0:
                if verbose:
                    print("Skipping matrix {0}...".format(name))
                o2.skip_op2_matrix(trailer)
            else:
                if name.find('USET') == 0:
                    if verbose:
                        print("Reading USET table {0}...".format(name))
                    uset = o2._read_op2_uset()
                    continue
                else:
                    if verbose:
                        print("Skipping table %r..." % name)
                o2.skip_op2_table()

    return {'uset': uset}

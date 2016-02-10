%---------------------------------------------------------------------------------------------------
%   File      :  read_op2.m
%   Author    :  Jörn Biedermann
%   Purpose   :  read nastran op2 file
%
%
%
%                                             Copyright (c) 2013 by Jörn Biedermann           _ /|_
%                                                           German Aerospace Center (DLR)    /_/_/_/
%                                                           Institute of Aeroelasticity        |/
%---------------------------------------------------------------------------------------------------
% clear all
% clc
% close all
% arch='n';
% fid = fopen('mk_plate_lumped.op2','r');
% mcs_op2read('mk_plate_lumped.op2','n','GEOM1')
function varout=read_op2(file)
fid = fopen(file,'r');
%% read Nastran Header
%get key
key = getkey(fid);
% key = 3 Nastran Header
if key==3
   recstart(fid)
   % date
   date = fread(fid,key,'int');
   recend(fid)
   % key=7
   key = getkey(fid);
   % Nastran Header
   recstart(fid)
   nheader = char(fread(fid,key*4,'char').');
   recend(fid)
   % key=2
   key = getkey(fid);
   % label
   recstart(fid)
   label = char(fread(fid,key*4,'char').');
   recend(fid)
   %key=-1
   key = getkey(fid);
   % key=-1 OP2NEW=0 or key=3 OP2NEW=1
   if key==-1
      %key=0
      key = getkey(fid);
   elseif key==3
      % Nastran Program Version
      recstart(fid)
      npv= char(fread(fid,key*4,'char').');
      recend(fid)
      %key=-1
      key = getkey(fid);
      %key=0
      key = getkey(fid);
   end
   
   % key = 2 no Nastran Header
elseif key==2
   frewind(fid);
end
%% read data blocks
EOD=-1;
while EOD~=0
   [ dbn,nt,dbn2,dbn2_data,npv2,lrt,data,matrix,ndata,EOD] = read_db(fid,EOD);
   if EOD~=0
      disp(['reading .... ',dbn]);
      eval([dbn,'.nt=nt;']);
      eval([dbn,'.dbn2=dbn2;']);
      eval([dbn,'.dbn2_data=dbn2_data;']);
      eval([dbn,'.npv2=npv2;']);
      eval([dbn,'.lrt=lrt;']);
      eval([dbn,'.data=data;']);
      eval([dbn,'.ndata=ndata;']);
      eval([dbn,'.matrix=matrix;']);
      eval(['varout.',dbn,'=',dbn,';']);
      clear dbn nt dbn2 lrt4 data ndata matrix
   end
end
fclose(fid);
end

function [ dbn,nt,dbn2,dbn2_data,npv2,lrt,data,matrix,ndata,EOD] = read_db(fid,EOD)
%% read Nastran Tables and Matrices
% key=2 or key=0 || isempty(key)==1 end of data
try
   key = getkey(fid);
   % Data Block Name
   recstart(fid)
   dbn = char(fread(fid,key*4,'char').');
   recend(fid)
   % key=-1 / OP2NEW=0 or key=3 / OP2NEW=1
   key = getkey(fid);
   if key==-1
      % key=7
      key = getkey(fid);
      % nastran trailer
      recstart(fid)
      nt= fread(fid,key,'int');
      recend(fid)
      % key=-2
      key = getkey(fid);
      % key=1
      key = getkey(fid);
      % key=0 (logical record type == 0 => table, else matrix)
      key = getkey(fid);
      % key=2
      key = getkey(fid);
      % data block name and data
      recstart(fid);
      dbn2 = char(fread(fid,2*4,'char').');
      if key>2
         dbn2_data = fread(fid,4*(key-2),'uchar');
      else
         dbn2_data = [];
      end
      recend(fid)
      %key=-3
      key = getkey(fid);
      npv2=[];
   elseif key==3
      recstart(fid)
      npv2= char(fread(fid,key*4,'char').');
      recend(fid)
      %key=-1
      key = getkey(fid);
      %key=7
      key = getkey(fid);
      % nastran trailer
      recstart(fid)
      nt= fread(fid,key,'int');
      recend(fid)
      %key=-2
      key = getkey(fid);
      %key=1
      key = getkey(fid);
      %key=0
      key = getkey(fid);
      %key=2
      key = getkey(fid);
      % data block name and data
      recstart(fid);
      dbn2 = char(fread(fid,2*4,'char').');
      if key>2
         dbn2_data = fread(fid,4*(key-2),'uchar');
      else
         dbn2_data = [];
      end
      recend(fid)
      %key=-3
      key = getkey(fid);
   end
   %% read table or matrix
   EOF=-1;
   ndata=[];
   data=[];
   matrix=sparse(1);
   t2=1;
   t3=1;
   while EOF~=0
      %key=1
      key = getkey(fid);
      if key~=1
         EOF=0;
         return
      else
         % logical record 4 of table
         recstart(fid);
         lrt = fread(fid,key,'int');
         recend(fid);
         if lrt==0
            %% Format for table
            %key=1>0
            key = getkey(fid);
            if key == 0 %% end of data
               EOF=0;
               break
            else
               t=1;
               temp=[];
               while key>0
                  if t==1
                     %data
                     ndata=[ndata key];
                  else
                     ndata(end)=ndata(end)+key;
                  end
                  recstart(fid);
%                   data=[data fread(fid,4*key,'*uchar').'];
                  temp=[temp fread(fid,4*key,'*uchar').'];
                  recend(fid);
                  %key<0 break while loop or key>0 add data
                  key = getkey(fid);
                  t=t+1;
               end
               data{t2}=temp;
               clear temp
               t2=t2+1;
            end
            matrix=[];
         else
            %% Format for matrix
            %key>0 Number of non zero terms in next string in word unit
            key = getkey(fid);
            while key>0
               recstart(fid);
               irow=fread(fid,1,'int');
               for t2=0:(key)/2-1
                  matrix(irow+t2,t3) = fread(fid,1,'double');
               end
               recend(fid);
               %key>0 Number of non zero terms in next string in word unit or
               %key < 0 end of colum
               key = getkey(fid);
            end
            t3=t3+1;
            data=[];
         end
      end
   end
catch
   EOD=0;
   dbn=[];
   nt=[];
   dbn2=[];
   dbn2_data=[];
   npv2=[];
   lrt=[];
   data=[];
   matrix=[];
   ndata=[];
   return
end
end
function key = getkey(fid)

recstart(fid);

key = fread(fid,1,'int');

recend(fid);

end
function recstart(fid)
fseek(fid,4,'cof');
end

function recend(fid)
fseek(fid,4,'cof');
end

%---------------------------------------------------------------------------------------------------
%   File      : sort_USET.m
%   Author    : J?rn Biedermann
%   Purpose   : sort USET table
%
%
%          =====================================================================
%          'USET'   - NASTRAN Set Information,
%                     written if enforced by HBResp DMAP,
%                     Each word corresponds to each degree of freedom in the
%                     g-set (in internal order) and contains ones in the speci-
%                     fied bit positions indicating the displacement sets to
%                     which the degree of freedom belongs.
%
%                     ti( 1,:) = [internal number of degree of freedom]
%                     ti( 2,:) = [set identification Number]
%
%          =====================================================================
%
%                                             Copyright (c) 2013 by J?rn Biedermann           _ /|_
%                                                           German Aerospace Center (DLR)    /_/_/_/
%                                                           Institute of Aeroelasticity        |/
%---------------------------------------------------------------------------------------------------
% BSP.:
% db=read_op2('E:\Kinetic_Energy_study\mk_plate_coupled.op2')
function table=sort_USET(db)
buffer1 = typecast(db.USET.data{1},'int32').';
buffer1 = dec2bin(buffer1,32);
buffer1 = cellstr(buffer1);
buffer1 = strfind(buffer1,'1');
buffer1 = cell2mat(buffer1);
% 
% for i = 1:size(buffer1,1)
%     if (buffer1{i} ~= 31) && (buffer1{i} ~= 32) %~
%         str = ['Found: ', num2str(buffer1{i})];
%         disp(str)
%     end
% end
% 
% buffer1 = cat(2,buffer1{:});
disp_set= cell(length(buffer1),1);
if any(buffer1==1)
   disp_set(find(buffer1==1)) = repmat(cellstr('U1'),length(find(buffer1==1)),1);
end
if any(buffer1==2)
   disp_set(find(buffer1==2)) = repmat(cellstr('U2'),length(find(buffer1==2)),1);
end
if any(buffer1==3)
   disp_set(find(buffer1==3)) = repmat(cellstr('U3'),length(find(buffer1==3)),1);
end
if any(buffer1==4)
   disp_set(find(buffer1==4)) = repmat(cellstr('U4'),length(find(buffer1==4)),1);
end
if any(buffer1==5)
   disp_set(find(buffer1==5)) = repmat(cellstr('U5'),length(find(buffer1==5)),1);
end
if any(buffer1==6)
   disp_set(find(buffer1==6)) = repmat(cellstr('U6'),length(find(buffer1==6)),1);
end
if any(buffer1==7)
   disp_set(find(buffer1==7)) = repmat(cellstr('V'),length(find(buffer1==7)),1);
end
if any(buffer1==8)
   disp_set(find(buffer1==8)) = repmat(cellstr('FR'),length(find(buffer1==8)),1);
end
if any(buffer1==9)
   disp_set(find(buffer1==9)) = repmat(cellstr('T'),length(find(buffer1==9)),1);
end
if any(buffer1==10)
   disp_set(find(buffer1==10)) = repmat(cellstr('Q'),length(find(buffer1==10)),1);
end
if any(buffer1==11)
   disp_set(find(buffer1==11)) = repmat(cellstr('B'),length(find(buffer1==11)),1);
end
if any(buffer1==12)
   disp_set(find(buffer1==12)) = repmat(cellstr('C'),length(find(buffer1==12)),1);
end
if any(buffer1==13)
   disp_set(find(buffer1==13)) = repmat(cellstr('PA'),length(find(buffer1==13)),1);
end
if any(buffer1==14)
   disp_set(find(buffer1==14)) = repmat(cellstr('K'),length(find(buffer1==14)),1);
end
if any(buffer1==15)
   disp_set(find(buffer1==15)) = repmat(cellstr('SA'),length(find(buffer1==15)),1);
end
if any(buffer1==16)
   disp_set(find(buffer1==16)) = repmat(cellstr('PS'),length(find(buffer1==16)),1);
end
if any(buffer1==17)
   disp_set(find(buffer1==17)) = repmat(cellstr('D'),length(find(buffer1==17)),1);
end
if any(buffer1==18)
   disp_set(find(buffer1==18)) = repmat(cellstr('FE'),length(find(buffer1==18)),1);
end
if any(buffer1==19)
   disp_set(find(buffer1==19)) = repmat(cellstr('NE'),length(find(buffer1==19)),1);
end
if any(buffer1==20)
   disp_set(find(buffer1==20)) = repmat(cellstr('P'),length(find(buffer1==20)),1);
end
if any(buffer1==21)
   disp_set(find(buffer1==21)) = repmat(cellstr('E'),length(find(buffer1==21)),1);
end
if any(buffer1==22)
   disp_set(find(buffer1==22)) = repmat(cellstr('SB'),length(find(buffer1==22)),1);
end
if any(buffer1==23)
   disp_set(find(buffer1==23)) = repmat(cellstr('SG'),length(find(buffer1==23)),1);
end
if any(buffer1==24)
   disp_set(find(buffer1==24)) = repmat(cellstr('L'),length(find(buffer1==24)),1);
end
if any(buffer1==25)
   disp_set(find(buffer1==25)) = repmat(cellstr('A'),length(find(buffer1==25)),1);
end
if any(buffer1==26)
   disp_set(find(buffer1==26)) = repmat(cellstr('F'),length(find(buffer1==26)),1);
end
if any(buffer1==27)
   disp_set(find(buffer1==27)) = repmat(cellstr('N'),length(find(buffer1==27)),1);
end
if any(buffer1==28)
   disp_set(find(buffer1==28)) = repmat(cellstr('G'),length(find(buffer1==28)),1);
end
if any(buffer1==29)
   disp_set(find(buffer1==29)) = repmat(cellstr('R'),length(find(buffer1==29)),1);
end
if any(buffer1==30)
   disp_set(find(buffer1==30)) = repmat(cellstr('O'),length(find(buffer1==30)),1);
end
if any(buffer1==31)
   disp_set(find(buffer1==31)) = repmat(cellstr('S'),length(find(buffer1==31)),1);
end
if any(buffer1==32)
   disp_set(find(buffer1==32)) = repmat(cellstr('M'),length(find(buffer1==32)),1);
end

table.disp_set=disp_set;
table.bitpos=buffer1;

end
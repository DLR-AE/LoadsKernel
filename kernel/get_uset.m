function out = get_uset(filename)

dbOP2 = read_op2([filename]);
USET = sort_USET(dbOP2);

out = USET;

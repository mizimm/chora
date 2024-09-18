function [x,y,z,vx,vy,vz,ex,ey,ez,spid] = readbin(filename)

fid=fopen(filename,'rb');
np=fread(fid,1,'int');

x=fread(fid,np,'double');
y=fread(fid,np,'double');
z=fread(fid,np,'double');
vx=fread(fid,np,'double');
vy=fread(fid,np,'double');
vz=fread(fid,np,'double');
ex=fread(fid,np,'double');
ey=fread(fid,np,'double');
ez=fread(fid,np,'double');
%phi=fread(fid,np,'double');
spid=fread(fid,np,'int');
fclose(fid);

%piano.m
clear all
nmax= 5;
tnote = [0.5, 1.0, 1.5, 2.0, 2.5]; %array of onset times of the notes (s)
dnote = [0.4, 0.4, 0.4, 0.4, 0.4]; %array of durations of the notes (s)
anote = [0.7, 0.8, 0.9, 0.8, 0.7]; %array of relative amplitudes of the notes
inote = [1, 5, 8, 5, 1]; %array of string indices of the notes

L=1;
J=81; dx=L/(J-1);
flow = 220;
nstrings = 25;

% moisture parameter
moisture = 0.1;

% temperature and pedal parameters
temp = 85;
pedal_pressed = false;

% Calculate temperature effect
% Assuming a linear change in frequency with temperature: 0.1% per degree Celsius deviation from 20°C
temp_effect = 1 - 0.001 * (temp - 20);

for i=1:nstrings
%     for tension instead?
    f(i) = flow * 2^((i-1)/12) * temp_effect;
    tau(i) = 1.2 * (440 / f(i)) * (1 - 0.5 * moisture); % decay time for moisture
    M(i) = 1;
    T(i) = M(i) * (2 * L * f(i))^2 * (1 - 0.5 * moisture);
    R(i) = (2 * M(i) * L^2) / (tau(i) * pi^2);
    % Find the largest stable timestep for string i
    dtmax(i) = -R(i) / T(i) + sqrt((R(i) / T(i))^2 + dx^2 / (T(i) / M(i)));
end

if pedal_pressed
    tmax_extension = 10;
else
    tmax_extension = 0;
end

dtmaxmin = min(dtmax);
nskip = ceil(1/(8192*dtmaxmin));
dt=1/(8192*nskip);

tmax=tnote(nmax)+dnote(nmax)+tmax_extension;
clockmax=ceil(tmax/dt);

tstop=zeros(nstrings,1);
H=zeros(nstrings,J);
V=zeros(nstrings,J);

xh1=0.25*L;xh2=0.35*L;

jstrike=ceil(1+xh1/dx):floor(1+xh2/dx);
j=2:(J-1);

count=0;
S=zeros(1,ceil(clockmax/nskip));
tsave = zeros(1,ceil(clockmax/nskip));
n=1;
for clock=1:clockmax
    t=clock*dt;
    while((n <= nmax) && (tnote(n) <= t))
        V(inote(n), jstrike) = anote(n);
        if pedal_pressed
            % If pedal is pressed, extend the decay significantly
            tstop(inote(n)) = t + dnote(n) + tmax_extension;
        else
            tstop(inote(n)) = t + dnote(n);
        end
        n = n + 1;
    end
    for i=1:nstrings
        if(t > tstop(i))
            H(i,:)=zeros(1,J);
            V(i,:)=zeros(1,J);
        else
            V(i,j)=V(i,j) +(dt/dx^2)*(T(i)/M(i))*(H(i,j+1)-2*H(i,j)+H(i,j-1)) +(dt/dx^2)*(R(i)/M(i))*(V(i,j+1)-2*V(i,j)+V(i,j-1));
            H(i,j)=H(i,j)+dt*V(i,j);
        end
    end
    if(mod(clock,nskip)==0)
        count=count+1;
        S(count)=sum(H(:,2));
        tsave(count)=t;
    end
end
soundsc(S(1:count));
plot(tsave(1:count),S(1:count));

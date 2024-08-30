% Parámetros
x_min = -5;
x_max = 5;
omega_min = -10;
omega_max = 10;
num_points = 500;

% Generar los valores de x y omega
x = linspace(x_min, x_max, num_points);
omega = linspace(omega_min, omega_max, num_points);

% Definir la función f(x)
f = @(x) sin(exp(cos(x)));

% Calcular la señal f(x)
signal = f(x);

% Calcular la Transformada de Fourier
fourier_result = zeros(size(omega));
for k = 1:length(omega)
    integrand = @(x) f(x) .* exp(-1i * omega(k) * x);
    fourier_result(k) = integral(integrand, x_min, x_max);
end

% Definir la ventana de Gabor g(t, tau)
g = @(t, tau, omega) (pi)^(-1 / 4) * exp(-((t - tau).^2) / 2) .* exp(1i * omega * (t - tau));

% Calcular la Transformada de Gabor
gabor_result = zeros(num_points, num_points);
for ti = 1:num_points
    t = x(ti);
    for wi = 1:num_points
        omega_i = omega(wi);
        integrand = @(tau) f(tau) .* g(t, tau, omega_i) .* exp(-1i * omega_i * tau);
        gabor_result(ti, wi) = integral(integrand, x_min, x_max);
    end
end

% Graficar la señal f(x)
figure;
plot(x, signal);
title('Señal en función del tiempo');
xlabel('x (tiempo)');
ylabel('f(x)');
grid on;

% Graficar la Transformada de Fourier
figure;
plot(omega, abs(fourier_result));
title('Transformada de Fourier de f(x)');
xlabel('ω (frecuencia)');
ylabel('|F(ω)|');
grid on;

% Graficar la Transformada de Gabor en 3D
[T, W] = meshgrid(x, omega);
figure;
surf(T, W, abs(gabor_result)');
shading interp; % Suaviza la coloración
title('Transformada de Gabor');
xlabel('t (tiempo)');
ylabel('ω (frecuencia)');
zlabel('|G(t, ω)|');
colorbar;
import numpy
import tools
import numpy.fft as fft
import matplotlib.pyplot as plt

class Gaussian:
    '''
    Источник, создающий гауссов импульс
    '''

    def __init__(self, dg, wg, eps=1.0, mu=1.0, Sc=1.0, magnitude=1.0):
        '''
        magnitude - максимальное значение в источнике;
        dg - коэффициент, задающий начальную задержку гауссова импульса;
        wg - коэффициент, задающий ширину гауссова импульса.
        '''
        self.dg = dg
        self.wg = wg
        self.eps = eps
        self.mu = mu
        self.Sc = Sc
        self.magnitude = magnitude

    def getField(self, m, q):
        e = (q - m * numpy.sqrt(self.eps * self.mu) / self.Sc - self.dg) / self.wg
        return self.magnitude * numpy.exp(-(e ** 2))

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    #Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 1700

    #Размер области моделирования в метрах
    X = 1

    #Размер ячейки разбиения
    dx = 1e-3

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = 200

    # Датчики для регистрации поля
    probesPos = [175,225]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    #1й слой диэлектрика
    eps1 = 1.1
    d1 = 0.03
    layer_1 = int(maxSize / 2) + int(d1 / dx)

    #2й слой диэлектрика
    eps2 = 2.2
    d2 = 0.04
    layer_2 = layer_1 + int(d2 / dx)

    #3й слой диэлектрика
    eps3 = 4.0
    d3 = 0.06
    layer_3 = layer_2 + int(d3 / dx)

    #4й слой диэлектрика
    eps4 = 4.5

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[int(maxSize/2):layer_1] = eps1
    eps[layer_1:layer_2] = eps2
    eps[layer_2:layer_3] = eps3
    eps[layer_3:] = eps4
    
    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    # Где начинается поглощающий диэлектрик слева
    layer_loss_x_left = 50

    # Где начинается поглощающий диэлектрик справа
    layer_loss_x_right = 950

    # Потери в среде. Loss = sigma * dt / (2 * eps * eps0)
    loss = numpy.zeros(maxSize)
    loss[layer_loss_x_right:] = 0.02
    loss[:layer_loss_x_left] = 0.02

    # Коэффициенты для расчета поля Е
    ceze = (1 - loss) / (1 + loss)
    cezh = W0 / (eps * (1 + loss))

    # Коэффициенты для расчеты поля Н
    chyh = (1 - loss) / (1 + loss)
    chye = 1 / (W0 * (1 + loss))

    # Усреднение коэффициентов на границе поглощающего слоя
    ceze[layer_loss_x_left] = (ceze[layer_loss_x_left - 1]
                               + ceze[layer_loss_x_left + 1]) / 2
    cezh[layer_loss_x_left] = (cezh[layer_loss_x_left - 1]
                          + cezh[layer_loss_x_left + 1]) / 2

    ceze[layer_loss_x_right] = (ceze[layer_loss_x_right - 1]
                               + ceze[layer_loss_x_right + 1]) / 2
    cezh[layer_loss_x_right] = (cezh[layer_loss_x_right - 1]
                          + cezh[layer_loss_x_right + 1]) / 2

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    # Параметры гауссова импульса
    A0 = 100    # уровень ослабления в момент времени t=0
    Amax = 100  # Уровень ослабления спектра сигнала на частоте Fmax
    Fmax = 10e9 # заданная максимальная частота
    
    wg = numpy.sqrt(numpy.log(Amax)) / (numpy.pi * Fmax)
    dg = wg * numpy.sqrt(numpy.log(A0))

    wg = wg / dt
    dg = dg / dt

    dg0 = 15    # Дополнительная задержка

    source = Gaussian(dg + dg0, wg, eps[sourcePos], mu[sourcePos])
    
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(int(maxSize / 2))
    display.drawBoundary(layer_1)
    display.drawBoundary(layer_2)
    display.drawBoundary(layer_3)

    for q in range(maxTime):
        # Расчет компоненты поля H
        #Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Расчет компоненты поля Н
        Hy = chyh[:-1] * Hy + chye[:-1] * (Ez[1:] - Ez[:-1])

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getField(0, q)

        # Граничные условия для поля E
        Ez[0] = Ez[1]
        Ez[-1] = Ez[-2]

        # Расчет компоненты поля E
        #Hy_shift = Hy[:-1]
        #Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = ceze[1:-1] * Ez[1:-1] + cezh[1:-1] * (Hy[1:] - Hy_shift)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getField(-0.5, q + 0.5))

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 10 == 0:
            display.updateData(display_field, q)

    display.stop()
    

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # Максимальная и манимальная частоты для отображения
    # графика зависимости коэффициента отражения от частоты
    Fmin = 0e9
    Fmax = 10e9
    
    F = tools.SpectrumAndReflection(probes, dt, maxTime, Fmin, Fmax)

    # Отображение спектров падающего и отраженного сигнала
    F.Spectrum()
    # Отображение графика зависимости коэффициента отражения от частоты
    F.ReflectionCoefficient()


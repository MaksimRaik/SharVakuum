import numpy as np
import function as f
from timeit import *

res = open( 'resultN=' + str( f.N ) + '.txt', 'w'  )

start = default_timer()

f.start_values()

    #print( f.u )

Time = f.cycle_while(f.u, f.r, f.V, f.E, f.P, f.delta_t)

stop = default_timer()
total_time = stop - start

    # output running time in a nice format.
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)


Legend = [str(Time[i]) + 's' for i in np.arange(0, len( list(f.uy) ), 1)]  # легенда

        # график скорости
f.plot_graph(f.rx, f.uy, 'R', 'U', 'U(R)', Legend, False)

        # график плотности
f.plot_graph(f.rx, f.Rhoy, 'R', 'rho', 'rho(R)', Legend, True)

        # график энергии
f.plot_graph(f.rx, f.Ey, 'R', 'E', 'E(R)', Legend, True)

res.write( 'N =' + str(f.N) + ' time work =' + str( hours ) + 'h ' + str( mins ) + 'm ' + str( round(secs, 1) ) + 'sec ' + 'delta_t = ' + str(f.delta_t) + '\n' )

    #print( 'N=',f.N, 'time work='+str( hours ) + 'h ' + str( mins ) + 'm ' + str( round(secs, 1) ) + 'sec','delta_t=',dt )

res.close()

stop = default_timer()
total_time = stop - start

mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

print( 'time work='+str( hours ) + 'h ' + str( mins ) + 'm ' + str( round(secs, 1) ) + 'sec' )

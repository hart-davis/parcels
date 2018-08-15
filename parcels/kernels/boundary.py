def reflective_boundary(particle, fieldset, time, dt, interp_method='nearest'):
    
    # Calculate the distance in latitudinal direction (using 1.11e2 kilometer per degree latitude)
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
    # Calculate the distance in longitudinal direction, using cosine(latitude) - spherical earth
    lon_dist = (particle.lon - particle.prev_lon) * 1.11e2 * math.cos(particle.lat * math.pi / 180)
    # Calculate the total Euclidean distance travelled by the particle
    particle.distance = lon_dist + lat_dist
    
    """Not Necessery Only in My Case"""
    #particle.wet_dry = fieldset.wet_dry_grid[time, particle.lon, particle.lat, particle.depth]
    #prev_wet = fieldset.wet_dry_grid[time, particle.prev_lon, particle.prev_lat, particle.prev_dep]
    
    particle.prev_lon = particle.lon  # Set the stored values for next iteration.
    particle.prev_lat = particle.lat
    
    we = (particle.lon - particle.boun_lon) * 1.11e2 * math.cos(particle.lat * math.pi / 180)
    me = (particle.lat - particle.boun_lat) * 1.11e2 
    
    part_uposition = fieldset.U[time, particle.lon, particle.lat, particle.depth]
    part_vposition = fieldset.V[time, particle.lon, particle.lat, particle.depth]
    
    d_t = 0
    
    if part_uposition == 0 and part_vposition == 0:
        hyp = particle.distance
        opp = me
        adj = we

        tan_i = math.atan(opp/adj)

        tan_j = math.atan(opp/adj) 

        ang = tan_i * 57.2958
        if (ang>=0) and (ang<=90):    thh=90 * 0.0174533
        elif (ang>90) and (ang<=180): thh=-90 * 0.0174533
        elif (ang>180) and (ang<=270):thh=90 * 0.0174533
        elif (ang>270) and (ang<=360):thh=-90 * 0.0174533
        else: thh = 0
        my_ang = tan_i + thh
        #my_ang = random.uniform(my_ang+0.5, my_ang-0.5) #simplistic comparison of scattering

        mylon = hyp*math.sin(my_ang)
        mylat = hyp*math.cos(my_ang)
        
        particle.lat += -mylat / 1.11e2
        particle.lon += mylon / (1.11e2 * math.cos(particle.lat * math.pi / 180))
        particle.boun_int += 1

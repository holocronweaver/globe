#!/usr/bin/env python
from pyglet import image
from pyglet.gl import *
from pyglet.gl.glu import *
from pyglet.window import key
import math
import random
import time

"""
Find squared length of a vector.
"""
def length2(v):
    length2 = 0.0
    for x_i in v:
        length2 += x_i**2
    return length2

"""
Normalize 3D vector.
A normalized vector points in the same direction of the original
vector but has unit length (length = 1).
"""
def normalize(v):
    length = math.sqrt(length2(v))
    return [v[i] / length for i in range(3)]

"""
Add two 3D vectors.
"""
def add(u, v):
    return [u[i] + v[i] for i in range(3)]

"""
Subtract two 3D vectors.
"""
def sub(u, v):
    return [u[i] - v[i] for i in range(3)]

"""
Multiply a 3D vector by a scalar.
"""
def mul(s, v):
    return [s * v[i] for i in range(3)]

"""
Find cross product of two 3D vectors.
"""
def cross(u, v):
    c = [0, 0, 0]
    c[0] += u[1] * v[2] - v[1] * u[2]
    c[1] -= u[0] * v[2] - v[0] * u[2]
    c[2] += u[0] * v[1] - v[0] * u[1]
    return c

"""
Find dot product of two 3D vectors.
"""
def dot(u, v):
    c = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    return c

"""
Simple geometric plane.
"""
class Plane(object):
    #TODO add constructor for triangle, if useful
    def __init__(self, origin, normal):
        self.origin = origin
        self.normal = normal
        self.equation = normal + [-dot(normal,origin)]

    def is_front_facing_to(self, direction):
        return dot(self.normal, direction) <= 0

    # Signed distance to point.
    def distance_to(self, point):
        return dot(point, self.normal) + self.equation[3]


"""
Lazy gluSphere with collision handling.
"""
class Sphere(object):
    def __init__(self, radius=1, position=[0,0,0], velocity=[0,0,0],
                 visible=False, slices=0, stacks=0, texture=None):
        self.radius = radius
        self.position = position
        self.velocity = velocity

        self.visible = visible
        if self.visible:
            self.slices = slices
            self.stacks = stacks
            self.quadric = gluNewQuadric()
            self.texture = texture
            #gluQuadricDrawStyle(self.quadric, GLU_FILL)
            #gluQuadricTexture(self.quadric, GL_TRUE)
            #gluQuadricNormals(self.quadric, GLU_SMOOTH)
            #self.mysphereID = glGenLists(1)
            #glNewList(self.mysphereID, GL_COMPILE)
            if self.texture:
                self.image = image.load(self.texture)
                self.texture = self.image.get_texture()
                #gluQuadricTexture (self.quadric, GL_TRUE)

                glEnable(self.texture.target);
                glBindTexture(self.texture.target, self.texture.id);

                gluSphere(self.quadric,self.radius,self.slices,self.stacks);

                gluDeleteQuadric(self.quadric);
                glDisable(GL_TEXTURE_2D);


    # Check if point will be in sphere after moving.
    def collision_check_point(self, r, dr):
        r2 = 0
        for i in range(3):
            r2 += (r[i] + dr[i])**2
        if r2 <= self.radius**2:
            return True
        return False

    # Check if two spheres will overlap after moving.
    def collision_check_sphere(self, sphere, dt):
        r = add(sphere.position, mul(dt,sphere.velocity))
        distance_vector = sub(r, self.position)
        #d = sub( r, add(r,dr) )
        distance = math.sqrt(length2(distance_vector))
        #print 'checking',distance, self.radius, sphere.radius
        if distance <= self.radius + sphere.radius:
            return True
        return False

    """
    Respond to collision of sphere with another sphere.
    """
    def collision_response_sphere(self, sphere, dt):
        r_0 = sphere.position
        v_0 = sphere.velocity
        radius = sphere.radius

        # Bring incoming sphere as close as possible without
        # intersecting.
        hit = True
        t = dt
        t_best = 0

        for n in range(1, 5):
            piece = 1.0 / 2**n
            if hit: t -= piece
            else: t += piece
            test_sphere = Sphere(radius=radius, position=r_0, velocity=v_0)
            hit = self.collision_check_sphere(test_sphere, dt)
            #print r_0, mul(t,v_0), n, t, piece, hit
            if not hit: t_best = t

        r_1 = add(r_0, mul(t_best,v_0))

        # Find tangent plane at point of contact (normal will suffice
        # for spheres).
        #origin = intersection_point
        #origin = mul(self.radius,normalize(gr))
        #normal = normalize( sub(gr,origin) )
        normal = normalize(r_0)
        #tplain = Plane(origin, normal)

        # Project incoming sphere velocity to tangent plane (normal
        # projection will suffice).
        v_0N = mul(dot(v_0,normal), normal)

        # Apply restitution force as an impulse
        # restitution = coefficient of restitution * velocity along normal
        # new velocity = old velocity - restitution
        k = 1
        restitution = mul(k,v_0N)
        v = sub(v_0,restitution)

        #print 'bounce!',normal,dot(v_0,normal),v_0N,restitution,v_0,v
        #exit()

        # Move incoming sphere along new velocity vector.
        sphere.position = add(r_1, mul(dt-t_best, v))
        sphere.velocity = v
        return sphere

    """
    Respond to collision of sphere with another sphere.
    """
    def collision_response_sphere2(self, sphere, dr):

        # convert from Cartesian (R3) to ellipsoid space
        r_e_space = [sphere.position[i] / self.radius for i in range(3)]
        dr_e_space = [dr[i] / self.radius for i in range(3)]
        radius = sphere.radius

        # iterate until we have a final position
        self.collision_recursion_depth = 0
        r_final, dr_final = self.collide_with_world(r_e_space, dr_e_space, radius)

        # add gravity pull!?

        # convert final result back to Cartesian space
        r_final = mul(self.radius,r_final)

        #print r_final, dr_final
        exit()

        # move the entity!?

        return r_final, dr_final

    """
    Move as close as possible to the sphere, then slide across
    the surface.
    r and delta-r are in ellipsoid space.
    """
    def collide_with_world(self, r, dr, radius):
        unit_scale = 100 / 100
        very_close_distance = 0.005 * unit_scale

        if (self.collision_recursion_depth > 5):
            #print 'UHHHHH!!!!!'
            #print r, dr
            return r, dr
        if not self.collision_check_sphere(r, dr, radius):
            #print 'YO!!!!!!!!!!!'
            r_new = add(r,dr)
            #print r_new, dr
            return r_new, dr

        destination = add(r,dr)
        r_new = r
        nearest_distance = 0 #FIXME

        """
        Only update if we are not already very close,
        and if so only move close to intersection, not
        to the exact spot.
        """
        if (nearest_distance >= very_close_distance):
            dr_norm = normalize(dr)
            dr_tiny = mul(nearest_distance - very_close_distance, dr_norm)
            r_new = add(r,dr_tiny)
            intersection_point -= mul(very_close_distance, dr_norm)

        """
        Respond to point moving into sphere by keeping the
        non-intersecting part of the movement vector and correcting
        for the intersecting part.
        Taken from Strawberry Cow Bear.
        """
        hit = True
        t = 1
        t_best = 0

        # move as close as possible to the sphere
        for n in range(1, 5):
            piece = 1.0 / 2**n
            if hit: t -= piece
            else: t += piece
            hit = self.collision_check_sphere(r, mul(t,dr), radius)
            print mul(t,dr), n, t, piece, hit
            if not hit: t_best = t
        # 'good' portion of movement vector
        gdr = mul(t_best,dr)
        gr = add(r,gdr)
        # 'bad' portion of movement vector
        bdr = mul(1 - t_best,dr)
        bdr_length = math.sqrt( length2(bdr) )

        # get tangent (sliding) plane at new position
        ##TODO generalize for intersection points whose spheres
        ## are not at origin
        origin = intersection_point
        #origin = mul(self.radius,normalize(gr))
        normal = normalize( sub(gr,origin) )
        #normal = normalize( mul(2.0/self.radius**2, r) )
        #plane = normal + [-1*dot(normal,origin)]
        tangent_plane = Plane(origin, normal)

        # project the original velocity vector to the sliding plane
        # to obtain new destination point
        #distance_to_destination = dot(destination, normal) + plane[3]
        distance_to_destination = tangent_plane.distance_to(destination)
        destination_new = sub(destination, mul(distance_to_destination,normal))

        # make new velocity vector by subtracting intersection from
        # new destination point
        dr_new = sub(destination_new,origin)

        # recurse until nothing is hit (update position) or velocity
        # vector gets tiny
        if length2(dr_new) < very_close_distance**2:
            return gr, gdr
        self.collision_recursion_depth += 1
        return self.collide_with_world(r_new, dr_new, radius)

        """
        # create local surface coordinate system
        normal, T1, T2 = self.get_local_coord(r)
        # project 'bad' movement vector on local coordinates
        bdr_local = [dot(bdr,T1), dot(bdr,normal), dot(bdr,T2)]
        # disallow negative normal vector values
        if bdr_local[1] < 0: bdr_local[1] = 0
        # project back into global coordinates
        bdr_norm_proj = [bdr_local[1] * normal[i] for i in range(3)]
        bdr_T1_proj = [bdr_local[0] * T1[i] for i in range(3)]
        bdr_T2_proj = [bdr_local[2] * T2[i] for i in range(3)]
        bdr_projected = [bdr_norm_proj[i]+bdr_T1_proj[i]+bdr_T2_proj[i] for i in range(3)]

        return r, [gdr[i] + bdr_projected[i] for i in range(3)]
        """

        """
        # get normalized tangent at moment of collision
        #TODO 3d-ify by taking cross product of d and v1, then normalize
        #tangent = [gr[1], -gr[0], 0]
        print bdr, r
        tangent = cross(bdr, r)
        tangent_length = length2(tangent)
        print tangent, tangent_length
        tangent = [tangent[i] / tangent_length for i in range(3)]
        # slide the bad vector on by
        u = bdr[0] * tangent[0] + bdr[1] * tangent[1] + bdr[2] * tangent[2]
        if u < -1E-8: gdr += [-tangent[i] * bdr_length for i in range(3)]
        else: gdr += [tangent[i] * bdr_length for i in range(3)]
        #r = [r[i] + gdr[i] for i in range(3)]
        return r, gdr + bdr_projected
        """

    def draw(self):
        if self.texture:
            glEnable(self.texture.target)
            glBindTexture(self.texture.target, self.texture.id)
            glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
            glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
            glEnable(GL_TEXTURE_GEN_S)
            glEnable(GL_TEXTURE_GEN_T)

            rotate = 90
            glRotatef(180,1.0,0.0,0.0);
            glRotatef(rotate,0.0,0.0,1.0);
            gluQuadricTexture(self.quadric,1);

        gluSphere(self.quadric,self.radius,self.slices,self.stacks)

        if self.texture:
            glDisable(GL_TEXTURE_GEN_S)
            glDisable(GL_TEXTURE_GEN_T)
            glDisable(self.texture.target)




"""
Contains the main game loop, window management, input/output handling,
first-person camera control, collision detection and response,
and physics (gravity).
"""
class Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.exclusive = False
        self.flying = False
        # movement on (x,z) plane
        self.strafe = [0, 0]
        # rotation angles
        # (x-z plane angle - from x-axis, y-axis angle - from x-z plane)
        self.rotation = (0, 0)
        # player walking speed
        self.walk_speed = 10
        self.reticle = None
        # jumping displacement
        #self.dr = [0,0,0]
        self.player = Sphere(radius=0.5, position=(60, 58, 43), visible=False)
        self.planet = Sphere(radius=32, #texture='earthmap.jpg',
                             slices=60, stacks=60, visible=True)
        self.label = pyglet.text.Label('', font_name='Arial', font_size=18,
            x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        pyglet.clock.schedule_interval(self.update, 1.0 / 60)

    def set_exclusive_mouse(self, exclusive):
        super(Window, self).set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def get_sight_vector(self):
        x, y = self.rotation # phi, theta
        m = math.cos(math.radians(y)) # x-
        dy = math.sin(math.radians(y))
        dx = math.cos(math.radians(x - 90)) * m
        dz = math.sin(math.radians(x - 90)) * m
        return (dx, dy, dz)

    """
    Calculates walking velocity vector by treating current view
    as a new Cartesian axis (e.g. x,y,z) and allowing motion controls
    to move on the horizontal plane.
    """
    def get_walk_vector(self):
        if any(self.strafe):
            x, y = self.rotation
            strafe = math.degrees(math.atan2(*self.strafe))
            #if self.flying:
            m = math.cos(math.radians(y))
            dy = math.sin(math.radians(y))
            if self.strafe[1]:
                dy = 0.0
                m = 1
            if self.strafe[0] > 0:
                dy *= -1
            dx = math.cos(math.radians(x + strafe)) * m
            dz = math.sin(math.radians(x + strafe)) * m
            #else:
            #    dy = 0.0
            #    dx = math.cos(math.radians(x + strafe))
            #    dz = math.sin(math.radians(x + strafe))
        else:
            dy = 0.0
            dx = 0.0
            dz = 0.0
        return (dx, dy, dz)

    def get_distance_vector(self, point):
        d = [0,0,0]
        for i in range(3):
            d[i] = self.player.position[i] - point.position[i]
        return d

    # Game loop.
    def update(self, dt):
        m = 8
        dt = min(dt, 0.2)
        for _ in xrange(m):
            self._update(dt / m)

    def _update(self, dt):
        # walking
        walk_velocity_vector = mul(self.walk_speed, self.get_walk_vector())
        #print 'jive walkin:',walk_velocity_vector
        #self.player.velocity = add(self.player.velocity, walk_velocity_vector)
        self.player.velocity = walk_velocity_vector
        # gravity
        if not self.flying:
            #g = [-3E3 / x_i**2 for x_i in self.get_distance_vector(self.planet)]
            #vector_to_planet = self.get_distance_vector(self.planet)
            vector_to_planet = self.player.position
            r2 = length2(vector_to_planet)
            r_hat = normalize(vector_to_planet)
            g = [-6E6 / r2 * r_hat[i] for i in range(3)]
            #self.dr[i] -= g[i] * dt**2 / 2
            self.player.velocity = add(self.player.velocity, mul(dt, g))
            #self.player.velocity[0] += mul(dt,g)[0]
            #print 'gravitatin:',g,self.player.velocity
            # establish terminal velocity to simulate air friction
            #self.dr[i] = max(self.dr[i], -0.5 * sign)
        # collision
        player = self.collide_planet(self.player, dt)
        # move player
        self.player.position = add(self.player.position, mul(dt,self.player.velocity))
        # remove impulse vectors
        self.player.velocity = sub(self.player.velocity, walk_velocity_vector)

    """
    Collision detection and response for collision
    bounding sphere and planet.
    """
    def collide_planet(self, sphere, dt):
        hit = self.planet.collision_check_sphere(sphere, dt)
        if (hit):
            #print 'collision!'
            sphere = self.planet.collision_response_sphere(sphere, dt)
        return sphere

    def on_mouse_press(self, x, y, button, modifiers):
        if self.exclusive:
            vector = self.get_sight_vector()
            return
        else:
            self.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.exclusive:
            m = 0.15
            x, y = self.rotation
            x, y = x + dx * m, y + dy * m
            #y = max(-90, min(90, y))
            self.rotation = (x, y)

    def on_key_press(self, symbol, modifiers):
        if symbol in (key.W, key.UP):
            self.strafe[0] -= 1
        elif symbol in (key.S, key.DOWN):
            self.strafe[0] += 1
        elif symbol in (key.A, key.LEFT):
            self.strafe[1] -= 1
        elif symbol in (key.D, key.RIGHT):
            self.strafe[1] += 1
        elif symbol == key.SPACE:
            if self.dy == 0:
                self.dy = 0.065
        elif symbol in (key.LSHIFT, key.RSHIFT):
            self.speed += 30
        elif symbol in (key.LCTRL, key.RCTRL):
            self.speed -= 10
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol == key.TAB:
            self.flying = not self.flying
            self.speed = 15 if self.flying else 5

    def on_key_release(self, symbol, modifiers):
        if symbol in (key.W, key.UP):
            self.strafe[0] += 1
        elif symbol in (key.S, key.DOWN):
            self.strafe[0] -= 1
        elif symbol in (key.A, key.LEFT):
            self.strafe[1] += 1
        elif symbol in (key.D, key.RIGHT):
            self.strafe[1] -= 1
        elif symbol in (key.LSHIFT, key.RSHIFT):
            self.speed -= 30
        elif symbol in (key.LCTRL, key.RCTRL):
            self.speed += 10

    def on_resize(self, width, height):
        # label
        self.label.y = height - 10
        # reticle
        if self.reticle:
            self.reticle.delete()
        x, y = self.width / 2, self.height / 2
        n = 10
        self.reticle = pyglet.graphics.vertex_list(4,
            ('v2i', (x - n, y, x + n, y, x, y - n, x, y + n))
        )

    def set_2d(self):
        width, height = self.get_size()
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set_3d(self):
        width, height = self.get_size()
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65.0, width / float(height), 0.1, 60.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        x, y = self.rotation
        glRotatef(x, 0, 1, 0)
        glRotatef(-y, math.cos(math.radians(x)), 0, math.sin(math.radians(x)))
        x, y, z = self.player.position
        glTranslatef(-x, -y, -z)

    def on_draw(self):
        self.clear()
        self.set_3d()
        glColor3d(1, 1, 1)
        #self.model.batch.draw()
        self.planet.draw()
        self.set_2d()
        self.draw_label()
        #self.draw_reticle()

    def draw_label(self):
        x, y, z = self.player.position
        self.label.text = '%02d (%.2f, %.2f, %.2f)' % (
            pyglet.clock.get_fps(), x, y, z,)
        self.label.draw()

    """
    def draw_reticle(self):
        glColor3d(0, 0, 0)
        self.reticle.draw(GL_LINES)
    """

def setup_fog():
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, (c_float * 4)(0.53, 0.81, 0.98, 1))
    glHint(GL_FOG_HINT, GL_DONT_CARE)
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_DENSITY, 0.35)
    glFogf(GL_FOG_START, 20.0)
    glFogf(GL_FOG_END, 60.0)

def setup():
    glClearColor(0.53, 0.81, 0.98, 1)
    glEnable(GL_CULL_FACE)

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    #setup_fog()

def main():
    window = Window(width=1024, height=768, caption='Pyglet Sphere', resizable=True)
    window.set_exclusive_mouse(True)
    setup()
    pyglet.app.run()

if __name__ == '__main__':
    main()

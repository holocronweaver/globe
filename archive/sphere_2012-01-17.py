#!/usr/bin/env python
"""
Walk and jump on a spherical planet centered at the origin.
Uses pyglet to call OpenGL fixed functions routines and gluSphere.
"""
from pyglet import image
from pyglet.gl import *
from pyglet.gl.glu import *
from pyglet.window import key
from ctypes import c_float
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

def length(v):
    return math.sqrt(length2(v))

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
Multiply two matrices (stored as lists of lists).
"""
def matrix_mul(A, B):
    AB = [[None for x in range( len(B[0]) )] for y in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            AB[i][j] = 0
            for r in range(len(B)):
                AB[i][j] += A[i][r] * B[r][j]
    if len(AB) == 1:
        AB = AB[0]
    return AB

"""
Create a rotation matrix (3x3) for a given angle
about an arbitrary axis.
"""
def create_rotation_matrix(axis, angle):
    x, y, z = normalize(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation_matrix = [0,0,0]
    rotation_matrix[0] = [
        cos_a +  x**2 * (1 - cos_a),
        x * y * (1 - cos_a) - z * sin_a,
        x * z * (1 - cos_a) + y * sin_a,
    ]
    rotation_matrix[1] = [
        y * x * (1 - cos_a) + z * sin_a,
        cos_a + y**2 * (1 - cos_a),
        y * z * (1 - cos_a) - x * sin_a,
    ]
    rotation_matrix[2] = [
        z * x * (1 - cos_a) - y * sin_a,
        z * y * (1 - cos_a) + x * sin_a,
        cos_a + z**2 * (1 - cos_a),
    ]
    return rotation_matrix

class Camera(object):
    def __init__(self, position = [0,0,0],
                 look = [0,0,1], up = [0,1,0], right = [1,0,0],
                 mode='UVN'):
        self.position = position
        self.look = look
        self.up = up
        self.right = right
        self.mode = mode # currently unused

    def build_view_matrix(self):
        # Keep camera axes orthogonal.
        self.look = normalize(self.look)
        self.up = cross(self.look, self.right)
        self.right = cross(self.up, self.look)
        self.right = normalize(self.right)
        # Build the view matrix.
        '''
        matrix = [
            (self.right[0], self.up[0], self.look[0], 0),
            (self.right[1], self.up[1], self.look[1], 0),
            (self.right[2], self.up[2], self.look[2], 0),
            (self.position[0], self.position[1], self.position[2], 1)
        ]
        '''
        matrix = (GLfloat * 16)(
            self.right[0], self.up[0], self.look[0], 0,
            self.right[1], self.up[1], self.look[1], 0,
            self.right[2], self.up[2], self.look[2], 0,
            self.position[0], self.position[1], self.position[2], 1,
        )
        return matrix

    #def move(self, distance, direction = self.look):
    #    dr = mul(direction, distance)
    #    self.position = add(self.position, dr)

    def move(self, displacement_vector):
        self.position = add(self.position, displacement_vector)

    def pitch(self, angle):
        T = create_rotation_matrix(self.right, angle)
        print self.up, T
        self.up = matrix_mul([self.up], T)
        self.look = matrix_mul([self.look], T)

    def roll(self, angle):
        T = create_rotation_matrix(self.look, angle)
        self.right = matrix_mul([self.right], T)
        self.up = matrix_mul([self.up], T)

    def yaw(self, angle):
        T = create_rotation_matrix(self.up, angle)
        self.right = matrix_mul([self.right], T)
        self.look = matrix_mul([self.look], T)


"""
Simple geometric 2D plane.
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
        #self.look_at = #TODO!
        self.velocity = velocity

        self.visible = visible
        self.slices = slices
        self.stacks = stacks
        self.texture = texture

        self.quadric = gluNewQuadric()

        if self.texture:
            self.image = image.load(self.texture)
            self.texture = self.image.get_texture()
            gluQuadricTexture (self.quadric, GL_TRUE)

            glEnable(self.texture.target);
            glBindTexture(self.texture.target, self.texture.id);

            #gluSphere(self.quadric,self.radius,self.slices,self.stacks);

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

        # Move incoming sphere along new velocity vector.
        sphere.position = add(r_1, mul(dt-t_best, v))
        sphere.velocity = v
        return sphere

    def draw(self):
        if self.texture:
            glEnable(self.texture.target)
            glBindTexture(self.texture.target, self.texture.id)
            """
            glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
            glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
            glEnable(GL_TEXTURE_GEN_S)
            glEnable(GL_TEXTURE_GEN_T)
            """
            rotate = 90
            glRotatef(180,1.0,0.0,0.0)
            glRotatef(rotate,0.0,0.0,1.0)
            gluQuadricTexture(self.quadric,1)

        gluSphere(self.quadric,self.radius,self.slices,self.stacks)

        if self.texture:
            glDisable(GL_TEXTURE_GEN_S)
            glDisable(GL_TEXTURE_GEN_T)
            glDisable(self.texture.target)

    """
    Transform position and look_at vectors together so that they
    remain perpendicular.
    """
    def set_position(self, position):
        self.position = position



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
        self.walking = False
        self.jumping = False
        # movement on (x,z) plane
        self.strafe = [0, 0]
        # rotation angles
        # (x-z plane angle - from x-axis, y-axis angle - from x-z plane)
        #self.rotation = (0, 0)
        # rotation axis (pitch, yaw, roll)
        self.rotation = [0, 0, 0]
        # movement vectors
        self.walk_speed = 10
        self.walk_velocity = [0, 0, 0]
        self.jump_speed = 25
        self.jump_velocity = [0, 0, 0]
        self.walk_before_jump = [0, 0, 0]
        self.jump_time = 0
        self.gravity_velocity = [0, 0, 0]

        self.balls = []
        self.ball_speed = 0.5
        self.reticle = None
        # jumping displacement
        #self.dr = [0,0,0]
        self.player = Sphere(radius=1, position=(60, 58, 43), visible=False)
        self.planet = Sphere(radius=32, #texture='earthmap2.jpg',
                             slices=60, stacks=60, visible=True)
        self.planet2 = Sphere(radius=15, position=(40,40,40),
                              slices=60, stacks=60, visible=True)
        self.label = pyglet.text.Label('', font_name='Arial', font_size=18,
            x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        self.models = [self.player, self.planet, self.planet2]
        pyglet.clock.schedule_interval(self.update, 1.0 / 60)

        self.camera = Camera(position=self.player.position)

    def set_exclusive_mouse(self, exclusive):
        super(Window, self).set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    """
    def get_sight_vector(self):
        yaw, pitch = self.rotation # phi, theta
        m = math.cos(math.radians(pitch))
        dy = math.sin(math.radians(pitch))
        dx = math.cos(math.radians(yaw - 90)) * m
        dz = math.sin(math.radians(yaw - 90)) * m
        return (dx, dy, dz)
    """

    """
    Calculates direction of sight by treating current view
    as a new Cartesian axis (e.g. x,y,z) and allowing motion controls
    to move on the horizontal plane.
    """
    """
    def get_sight_vector(self):
        if any(self.strafe):
            roll, yaw, pitch = self.rotation
            strafe = math.degrees(math.atan2(*self.strafe))
            #if self.flying:
            m = math.cos(math.radians(pitch))
            dy = math.sin(math.radians(pitch))
            if self.strafe[1]:
                dy = 0.0
                m = 1
            if self.strafe[0] > 0:
                dy *= -1
            dx = math.cos(math.radians(yaw + strafe)) * m
            dz = math.sin(math.radians(yaw + strafe)) * m
            #else:
            #    dy = 0.0
            #    dx = math.cos(math.radians(x + strafe))
            #    dz = math.sin(math.radians(x + strafe))
            #FIXME convert dx,dy,dz from normal-tangent to world coordinates
        else:
            dy = 0.0
            dx = 0.0
            dz = 0.0
        return (dx, dy, dz)
    """

    def get_distance_vector(self, point):
        d = [self.player.position[i] - point.position[i] for i in range(3)]
        return d

    # Game loop.
    def update(self, dt):
        m = 8
        dt = min(dt, 0.2)
        for _ in xrange(m):
            self._update(dt / m)

    def _update(self, dt):
        self.player.velocity = [0,0,0]
        # walking
        if self.walking:
            self.walk_velocity = [0,0,0]
            #self.walk_velocity = mul(self.walk_speed, self.camera.look)
            #self.player.velocity = add(self.player.velocity, self.walk_velocity)
        else:
            self.walk_velocity = [0,0,0]
        # gravity
        if not self.flying:
            vector_to_planet = self.player.position
            r2 = length2(vector_to_planet)
            r_hat = normalize(vector_to_planet)
            g = [-6E4 / r2 * r_hat[i] for i in range(3)]
            self.gravity_velocity = add(self.gravity_velocity, mul(dt, g))
            # establish terminal velocity to simulate air friction
            #self.gravity_velocity = max(self.gravity_velocity, )
            self.player.velocity = add(self.player.velocity, self.gravity_velocity)
            #print 'gravitatin:',g,self.gravity_velocity,self.player.velocity
        if self.walking: # normal force
            self.gravity_velocity = [0,0,0]
        # jumping
        if self.jumping:
            if self.walking:
                self.walk_before_jump = self.walk_velocity
            sphere_normal = normalize(self.player.position)
            self.jump_velocity = mul(self.jump_speed, sphere_normal)
            self.player.velocity = add(self.player.velocity, self.jump_velocity)
            self.jump_time += dt
            if self.jump_time > 1:
                self.jumping = False
                self.jump_time = 0
                self.jump_velocity = [0,0,0]
                self.walk_before_jump = [0,0,0]
        if not self.walking:
            self.player.velocity = add(self.player.velocity, self.walk_before_jump)
        # collision
        self.collide_planet(self.player, dt)
        # move player
        self.player.position = add(self.player.position, mul(dt,self.player.velocity))

        # fly balls
        for ball in self.balls:
            ball.position = add(ball.position, mul(dt,ball.velocity))

    """
    Collision detection and response for collision
    bounding sphere and planet.
    """
    def collide_planet(self, sphere, dt):
        hit = self.planet.collision_check_sphere(sphere, dt)
        if (hit):
            self.walking = True
            #print 'collision!'
            sphere = self.planet.collision_response_sphere(sphere, dt)
        else:
            self.walking = False
        return sphere

    def on_mouse_press(self, x, y, button, modifiers):
        return
        if self.exclusive:
            sight_vector = self.get_sight_vector()
            ball = Sphere(radius=2, position=self.player.position,
                          velocity=mul(self.ball_speed, sight_vector),
                          visible=True, slices=10, stacks=10)
            self.balls.append(ball)
            self.models.append(ball)
            print self.balls[-1].velocity
            return
        else:
            self.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.exclusive:
            m = 0.15
            pitch, yaw, roll = self.rotation
            yaw, pitch = yaw + dx * m, pitch + dy * m
            #if not (-90 < pitch < 90):
            #    pitch = max(-90, min(80, pitch))
            #    dy = 0
            #TODO roll could come via joystick twist or mousewheel
            roll = 0
            self.rotation = (roll, yaw, pitch)

            """
            roll_axis = self.rotation_matrix[0]
            yaw_axis = self.rotation_matrix[1]
            pitch_axis = self.rotation_matrix[2]

            rotation_axis = add(mul(0, roll_axis),
                                 add( mul(dx * m, yaw_axis),
                                      mul(dy * m, pitch_axis) )
                             )
            print dx, dy
            rotation_angle = length(rotation_axis)
            rotation_axis = [rotation_axis[i] / rotation_angle for i in range(3)]
            print rotation_angle, rotation_axis

            rotation_matrix = create_rotation(rotation_axis, rotation_angle)
            #print rotation_matrix; exit()
            self.rotation_matrix = matrix_mul(rotation_matrix,self.rotation_matrix)

            self.rotation_axis = rotation_axis
            self.rotation_angle = rotation_angle
            """

            """ old rotation about x and y-axis
            x, y = self.rotation
            x, y = x + dx * m, y + dy * m
            print 'xy:',x,y
            """
            if self.walking:
                self.walking
                """ third try - would work if 3D cartesian axis rotation
                    coordinate transform was known
                # convert angles to spherical coordinates on normal surface axis
                # spherical -> cartesian -> normal cartesian -> normal spherical
                cartesian = self.get_sight_vector()
                normal = self.player.position
                norm_length = math.sqrt(length2(normal))
                dm = math.atan(normal[2]/normal[0])
                dn = math.acos(normal[1]/norm_length)
                # limit camera rotation on normal axis
                # convert back to world axis
                # spherical -> cartesian -> cartesian -> spherical
                """

                """ second try - does not work
                # convert surface normal to spherical coordinates
                normal = self.player.position
                norm_length = math.sqrt(length2(normal))
                dm = math.atan(normal[2]/normal[0])
                dn = math.acos(normal[1]/norm_length)
                # find camera angles in terms of surface normal axis
                m, n = x - dm, y - dn
                print 'mn:',m,n,dm,dn
                # limit camera rotation on normal axis
                n = max(-90, min(90, n))
                # convert back to world axis
                x, y = m + dm, n + dn
                """

                """ first try - does not work
                # rotate world axis to find surface tangent plane angles
                normal = self.player.position
                norm_length = math.sqrt(length2(normal))
                xz_length = math.sqrt( length2([normal[0],0,normal[2]]) )
                print normal, xz_length, normal[0], norm_length, normal[1]
                dm = math.acos(abs(normal[0])/xz_length)
                dn = math.acos(abs(normal[1])/norm_length)
                m, n = x + dm, y + dn
                print 'mn:',m,n,dm,dn
                # limit tangent-plane rotation to looking at feet and sky
                n = max(-90, min(90, n))
                # convert back to world axis
                x, y = m - dm, n - dn
                """
            #self.rotation = (yaw, pitch)

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
            if self.walking:
                self.jumping = True
        elif symbol in (key.LSHIFT, key.RSHIFT):
            self.walk_speed += 30
        elif symbol in (key.LCTRL, key.RCTRL):
            self.walk_speed -= 10
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol == key.TAB:
            self.flying = not self.flying
            self.walk_speed = 15 if self.flying else 5

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
            self.walk_speed -= 30
        elif symbol in (key.LCTRL, key.RCTRL):
            self.walk_speed += 10

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

    def set_2d_draw_mode(self):
        # window scaling and projection
        width, height = self.get_size()
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set_3d_draw_mode(self):
        # window scaling and projection
        width, height = self.get_size()
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65.0, width / float(height), 0.1, 60.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # camera rotation
        pitch, yaw, roll = self.rotation
        #x, y, z = self.rotation_axis
        #glRotatef(self.rotation_angle, x, y, z)
        view_matrix = []
        if self.walking and not self.flying:
            #position = self.camera.position
            normal = normalize(sub(self.camera.position,self.planet.position))
            self.camera.pitch(pitch)
            self.camera.yaw(yaw)
            #self.camera.yaw(yaw, direction = normal) #FIXME!
            '''
            look = [
                -math.sin(math.radians(roll)),
                math.cos(math.radians(roll)),
                0,
            ]
            look = [0,0,1]
            '''
            # ax + by + cz = position**2
            # If (x,z) = (1,1), then
            #
            '''
            x, z = [1, 1]
            look = [x, (length2(position)-up[0]*x-up[2]*z)/up[1], z]
            '''
            '''
            look = self.camera[0]
            look = normalize(look)
            up = cross(look, right)
            right = cross(up, look)
            right = normalize(right)
            print look
            print up
            print right
            '''
            '''
            gluLookAt(
                position[0], position[1], position[2],
                look[0]+position[0], look[1]+position[1], look[2]+position[2],
                up[0], up[1], up[2],
            )
            '''
            #glRotatef(yaw, up[0], up[1], up[2])
            #glRotatef(pitch, right[0], right[1], right[2])
            #glRotatef(0, look[0], look[1], look[2])
        else:
            self.camera.pitch(pitch)
            self.camera.yaw(yaw)

            #norm = normalize(self.player.position)
            #glRotatef(yaw, norm[0], norm[1], norm[2])

            #glRotatef(yaw, 0, 1, 0)
            #glRotatef(-pitch,
            #          math.cos(math.radians(yaw)), 0, math.sin(math.radians(yaw)))
        self.rotation = [0,0,0]
        # camera position
        position = self.player.position
        self.camera.position = position
        #x, y, z = self.player.position
        #glTranslatef(-x, -y, -z)

        view_matrix = self.camera.build_view_matrix()
        '''
        print view_matrix
        glMultMatrixf(view_matrix)
        '''
        look = self.camera.look
        view = add(look,position)
        up = self.camera.up
        print dot(look,view)
        gluLookAt(
            position[0], position[1], position[2],
            view[0], view[1], view[2],
            up[0], up[1], up[2],
        )


    def on_draw(self):
        self.clear()
        self.set_3d_draw_mode()
        glColor3d(1, 1, 1)
        #self.model.batch.draw()
        for model in self.models:
            model.draw()
        for ball in self.balls:
            ball.draw()
        #self.planet.draw()
        self.set_2d_draw_mode()
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
    glFogf(GL_FOG_DENSITY, 0.15)
    glFogf(GL_FOG_START, 20.0)
    glFogf(GL_FOG_END, 180.0)

def setup():
    glClearColor(0.53, 0.81, 0.98, 1)
    glEnable(GL_CULL_FACE)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)

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

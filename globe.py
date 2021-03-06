#!/usr/bin/env python
"""
Walk, jump, and shoot colorful balls on a rotating globe.
Uses Pyglet to call OpenGL fixed function routines, including gluSphere.
TODO
* optimize OpenGL calls to increase max number of balls
"""
from mathlib import *
from pyglet import image
from pyglet.gl import *
from pyglet.gl.glu import *
from pyglet.window import key
import math
from random import random

class Player(object):
    # states
    flying = False
    walking = False
    jumping = False
    near_globe = False
    # movement on (right,look) plane
    strafe = [0, 0]
    # rotation angles (pitch, yaw, roll)
    total_rotation = [0,0,0]
    # movement vectors
    walk_speed = 15
    walk_velocity = [0, 0, 0]
    jump_speed = 25
    jump_velocity = [0, 0, 0]
    walk_before_jump = [0, 0, 0]
    jump_time = 0
    gravity_velocity = [0, 0, 0]

    def __init__(self, position=(0,0,0), rotation=(0,0,0), height=1,
                 texture=None):
        self.camera = Camera(position=position)
        self.model = Sphere(radius=height, position=position, visible=False)
        self.rotation = rotation

    @property
    def height(self):
        return self.model.radius
    @height.setter
    def height(self, height):
        self.model.radius = height
    @property
    def velocity(self):
        return self.model.velocity
    @velocity.setter
    def velocity(self, velocity):
        self.model.velocity = velocity
    @property
    def position(self):
        return self.camera.position
    @position.setter
    def position(self, position):
        self.camera.position = position
        self.model.position = position

    def rotate(self, globe):
        pitch, yaw, roll = self.rotation
        if (self.walking or self.jumping or self.near_globe) and not self.flying:
            # Orient camera on surface tangent plane.
            normal = normalize(sub(self.camera.position,globe.position))
            self.camera.orient(normal)

            # Approximate pitch limitations of human head.
            self.total_rotation = add(self.rotation, self.total_rotation)
            total_pitch, total_yaw, total_row = self.total_rotation
            total_pitch = min(60, max(-80, total_pitch))
            self.total_rotation[0] = total_pitch

            self.camera.yaw(-math.radians(yaw))
            self.camera.pitch(-math.radians(total_pitch))
        else:
            self.camera.yaw(-math.radians(yaw))
            self.camera.pitch(-math.radians(pitch))
        self.rotation = [0,0,0]
        self.camera.orthonormalize()

    def update(self, dt, globe):
        self.velocity = [0,0,0]
        vector_to_globe = sub(self.position,globe.position)
        # walking
        if self.walking or self.flying:
            normal = normalize(vector_to_globe)
            axis = self.camera.axis
            if self.walking: # walk on sphere surface tangent plane
                right, up, look = orthonormalize(orient(axis, normal))
                if globe.rotate: # centripetal friction
                    axis_of_rotation = [0,0,1]
                    angular_velocity = globe.rotate[0] / dt / 920
                    rotation_vector = mul(angular_velocity,axis_of_rotation)
                    centripetal_velocity = cross(rotation_vector,self.position)
                    self.velocity = add(self.velocity, centripetal_velocity)
            else: # fly about
                right, up, look = axis
            strafe_look = mul(self.walk_speed * self.strafe[0], look)
            strafe_right = mul(self.walk_speed * -self.strafe[1], right)
            self.walk_velocity = add(strafe_look, strafe_right)
            self.velocity = add(self.velocity, self.walk_velocity)
        else:
            self.walk_velocity = [0,0,0]
        # gravity
        if not self.flying:
            r2 = length2(vector_to_globe)
            r_hat = normalize(vector_to_globe)
            g = [-3E6 / r2 * r_hat[i] for i in range(3)]
            self.gravity_velocity = add(self.gravity_velocity, mul(dt, g))
            # establish terminal velocity to simulate air friction
            terminal_velocity = 90
            if length(self.gravity_velocity) > terminal_velocity:
                self.gravity_velocity = mul(terminal_velocity, normalize(self.gravity_velocity))
            self.velocity = add(self.velocity, self.gravity_velocity)
            #print 'gravitatin:',g,self.gravity_velocity,length(self.gravity_velocity)
        if self.walking: # normal force
            self.gravity_velocity = [0,0,0]
        # jumping
        if self.jumping:
            self.max_jump_time = 2
            if self.jump_time < 1 or not self.near_globe:
                normal = normalize(vector_to_globe)
                direction = 1 if self.jump_time < 2 else 0
                self.jump_velocity = mul(direction * self.jump_speed, normal)
                self.velocity = add(self.velocity, self.jump_velocity)
                self.jump_time += dt
                if not self.walking:
                    self.velocity = add(self.velocity, self.walk_before_jump)
                else:
                    self.walk_before_jump = self.walk_velocity
            else:
                self.jumping = False
                self.jump_time = 0
                self.jump_velocity = [0,0,0]
                self.walk_before_jump = [0,0,0]
        # collision
        self.collide_globe(globe, dt)
        # move player
        self.position = add(self.model.position, mul(dt,self.velocity))

        if length(sub(self.position,globe.position)) < (globe.radius + self.height + 2):
            self.near_globe = True
        else:
            self.near_globe = False

        self.rotate(globe)

    """
    Collision detection and response for collision
    bounding sphere and globe.
    """
    def collide_globe(self, globe, dt):
        hit = globe.collision_check_sphere(self.model, dt)
        if hit:
            self.walking = True
            self.model = globe.collision_response_sphere(self.model, dt)
        else:
            self.walking = False


class Camera(object):
    def __init__(self, position = [0,0,0],
                 look = [0,0,1], up = [0,1,0], right = [1,0,0],
                 mode='UVN'):
        self.position = position
        self.look = look
        self.up = up
        self.right = right
        self.axis = [self.right, self.up, self.look]
        self.mode = mode # currently unused

    @property
    def axis(self):
        return (self.right, self.up, self.look)
    @axis.setter
    def axis(self, axis):
        self.right, self.up, self.look = axis

    def view_matrix(self):
        matrix = view_matrix(self.position, self.axis)
        return matrix

    def move(self, displacement_vector):
        self.position = add(self.position, displacement_vector)

    # Orient camera 'up' along a normalized direction.
    def orient(self, normal):
        self.axis = orient(self.axis, normal)
        self.orthonormalize()

    # Prevents floating point errors from accumulating.
    def orthonormalize(self):
        self.axis = orthonormalize(self.axis)

    def pitch(self, angle):
        T = rotation_matrix_2D(angle)
        self.look, self.up = matrix_mul(T, [self.look,self.up])

    def roll(self, angle):
        T = rotation_matrix_2D(angle)
        self.up, self.right = matrix_mul(T, [self.up,self.right])

    def yaw(self, angle):
        T = rotation_matrix_2D(angle)
        self.right, self.look  = matrix_mul(T, [self.right,self.look])


# Simple geometric 2D plane.
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

# Lazy gluSphere with collision handling.
class Sphere(object):
    rotation = 0

    def __init__(self, radius=1, position=[0,0,0], velocity=[0,0,0],
                 rotate=None, visible=False,
                 slices=0, stacks=0, color=None, texture=None):
        self.radius = radius
        self.position = position
        self.velocity = velocity
        self.rotate = rotate

        self.visible = visible
        self.slices = slices
        self.stacks = stacks
        self.color = color
        self.texture = texture

        self.quadric = gluNewQuadric()

        if self.texture:
            self.image = image.load(self.texture)
            self.texture = self.image.get_texture()
            gluQuadricTexture(self.quadric, GL_TRUE)

            glEnable(self.texture.target)
            glBindTexture(self.texture.target, self.texture.id)

            glDisable(self.texture.target)

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
        distance = math.sqrt(length2(distance_vector))
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
            if not hit: t_best = t

        r_1 = add(r_0, mul(t_best,v_0))

        # Find tangent plane at point of contact (normal will suffice
        # for spheres).
        normal = normalize(r_0)
        #tplain = Plane(origin, normal)

        # Project incoming sphere velocity to tangent plane (normal
        # projection will suffice).
        v_0N = mul(dot(v_0,normal), normal)

        # Apply restitution force as an impulse.
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
        glPushMatrix()
        if self.texture:
            glEnable(self.texture.target)
            glBindTexture(self.texture.target, self.texture.id)
            gluQuadricTexture(self.quadric,GL_TRUE)
        if self.color:
            r, g, b = self.color
            glColor3f(r, g, b)
        glTranslatef(self.position[0], self.position[1], self.position[2]);
        if self.rotate:
            glRotatef(self.rotation, 0.0, 0.0, 1.0)
            self.rotation += self.rotate[0]
        gluSphere(self.quadric,self.radius,self.slices,self.stacks)
        if self.texture:
            gluQuadricTexture(self.quadric,GL_FALSE)
            glDisable(self.texture.target)
        glPopMatrix()

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

        self.balls = []
        self.ball_speed = 2
        self.balls_max = 22

        self.player = Player(height=2, position=(290, 50, 40))
        self.globe = Sphere(radius=256, rotate=[0.01,None],#rotate=[0.01,None],
                            texture='images/earth.jpg',
                             slices=32000, stacks=32000, visible=True)
        self.sun = Sphere(radius=15, #position=(300, 50, 40),
                           texture='images/sun.jpg',
                           slices=30, stacks=30, visible=True)
        #self.moon = Sphere(radius=15, position=(290,50,40),
        #                   texture='images/moon.jpg', rotate=[50,50],
        #                   slices=60, stacks=60, visible=True)
        self.models = [self.player.model, self.globe, self.sun]

        self.label = pyglet.text.Label('', font_name='Arial', font_size=18,
            x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        self.reticle = None
        pyglet.clock.schedule_interval(self.update, 1.0 / 60)

    def set_exclusive_mouse(self, exclusive):
        super(Window, self).set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    # Game loop.
    def update(self, dt):
        time_steps = 8
        dt = min(dt, 0.2)
        for step in xrange(time_steps):
            self.player.update(dt / time_steps, self.globe)
            self.balls_update(dt / time_steps, self.globe)

    def balls_update(self, dt, globe):
        for ball in self.balls:
            vector_to_globe = sub(ball.position, globe.position)
            # collision
            hit = self.globe.collision_check_sphere(ball, dt)
            if hit:
                ball = globe.collision_response_sphere(ball, dt)
            for other_ball in self.balls:
                if ball is other_ball: continue
                hit = other_ball.collision_check_sphere(ball, dt)
                if hit:
                    ball = ball.collision_response_sphere(ball, dt)
            hit = self.player.model.collision_check_sphere(ball, dt)
            if hit:
                ball = self.player.model.collision_response_sphere(ball, dt)
            ball.position = add(ball.position, mul(dt,ball.velocity))

    def on_mouse_press(self, x, y, button, modifiers):
        if self.exclusive:
            if len(self.balls) == self.balls_max: return
            sight_vector = self.player.camera.look
            ball = Sphere(radius=2, position=add(self.player.position,mul(3,self.player.camera.look)),
                          velocity=mul(self.ball_speed, sight_vector),
                          color=[random(),random(),random()],
                          visible=True, slices=40, stacks=40)
            self.balls.append(ball)
        else:
            self.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.exclusive:
            rate = 0.015
            dx, dy = math.degrees(dx), math.degrees(dy)
            pitch, yaw, roll = self.player.rotation
            yaw, pitch = yaw + dx * rate, pitch + dy * rate
            roll = 0
            self.player.rotation = (pitch, yaw, roll)

    def on_key_press(self, symbol, modifiers):
        if symbol in (key.W, key.UP):
            self.player.strafe[0] += 1
        elif symbol in (key.S, key.DOWN):
            self.player.strafe[0] -= 1
        elif symbol in (key.D, key.RIGHT):
            self.player.strafe[1] += 1
        elif symbol in (key.A, key.LEFT):
            self.player.strafe[1] -= 1
        elif symbol == key.SPACE:
            if self.player.walking:
                self.player.jumping = True
        elif symbol in (key.LSHIFT, key.RSHIFT):
            self.player.walk_speed += 30
        elif symbol in (key.LCTRL, key.RCTRL):
            self.player.walk_speed -= 10
        elif symbol == key.TAB:
            self.player.flying = not self.player.flying
            self.player.walk_speed = 30 if self.player.flying else 15
            if self.player.flying: self.player.jumping = False
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)

    def on_key_release(self, symbol, modifiers):
        if symbol in (key.W, key.UP):
            self.player.strafe[0] -= 1
        elif symbol in (key.S, key.DOWN):
            self.player.strafe[0] += 1
        elif symbol in (key.D, key.RIGHT):
            self.player.strafe[1] -= 1
        elif symbol in (key.A, key.LEFT):
            self.player.strafe[1] += 1
        elif symbol in (key.LSHIFT, key.RSHIFT):
            self.player.walk_speed -= 30
        elif symbol in (key.LCTRL, key.RCTRL):
            self.player.walk_speed += 10

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
        width, height = self.get_size()
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set_3d_draw_mode(self):
        width, height = self.get_size()
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65.0, width / float(height), 0.1, 60.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def on_draw(self):
        self.clear()
        self.set_3d_draw_mode()

        position = self.player.position
        right, up, look = self.player.camera.axis
        view = add(look,position)
        gluLookAt(
            position[0], position[1], position[2],
            view[0], view[1], view[2],
            up[0], up[1], up[2],
        )

        glColor3d(1, 1, 1)
        for model in self.models:
            model.draw()
        for ball in self.balls:
            ball.draw()
        self.set_2d_draw_mode()
        self.draw_label()
        #self.draw_reticle()

        glFlush()

    def draw_label(self):
        x, y, z = self.player.position
        self.label.text = '%02d (%.2f, %.2f, %.2f)' % (
            pyglet.clock.get_fps(), x, y, z,)
        self.label.draw()

    def draw_reticle(self):
        glColor3d(0, 0, 0)
        self.reticle.draw(GL_LINES)


def setup_fog():
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, (c_float * 4)(0.53, 0.81, 0.98, 1))
    glHint(GL_FOG_HINT, GL_DONT_CARE)
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_DENSITY, 0.15)
    glFogf(GL_FOG_START, 10.0)
    glFogf(GL_FOG_END, 90.0)

def setup():
    glClearColor(0.53, 0.81, 0.98, 1)
    glEnable(GL_CULL_FACE)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    #glEnable(GL_POINT_SMOOTH)
    #glEnable(GL_LINE_SMOOTH)
    #glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
    #glBlendFunc(GL_SRC_ALPHA_SATURATE, GL_ONE);
    #glEnable(GL_BLEND);
    glEnable(GL_POLYGON_SMOOTH)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);


    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    setup_fog()

def main():
    window = Window(width=1024, height=768,
                    caption='Pyglet Sphere', resizable=True)
    window.set_exclusive_mouse(True)
    setup()
    pyglet.app.run()

if __name__ == '__main__':
    main()

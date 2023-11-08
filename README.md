# Physics-Informed Neural Network 

Learning how to implement a PINN using TensorFlow

## Examples
### 2D heat equation
$$
\dfrac{\partial T}{\partial t} = \alpha \nabla^2 T.
$$

### 2D temperature equation

$$
\dfrac{\partial T}{\partial t} + \mathbf{u}\cdot\nabla T = \alpha \nabla^2 T + S(T).
$$

### 2D cavity flow

$$
\begin{split}
    \nabla\cdot\mathbf{u}&=0 \\
    \dfrac{\partial \mathbf{u}}{\partial t} + 
    \left(\mathbf{u}\cdot\nabla\right)\mathbf{u} &=
    -\dfrac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u}
\end{split}
$$


### 2D shallow-water equations

$$
\begin{aligned}
    \dfrac{\partial h}{\partial t} & + \frac{\partial}{\partial x}\Bigl((H+h)u\Bigr)
        + \frac{\partial }{\partial y}\Bigl((H+h)v\Bigr)=0 \\
    \dfrac{\partial u}{\partial t} & + u\frac{\partial u}{\partial x} 
        + v\frac {\partial u}{\partial y} - fv = -g\frac {\partial h}{\partial x}
        - ku + \nu \left(\dfrac{\partial^2u}{\partial x^2}
        + \dfrac{\partial ^{2}u}{\partial y^2}\right),\\
    \dfrac{\partial v}{\partial t} & +u\dfrac {\partial v}{\partial x} 
        + v\dfrac {\partial v}{\partial y} + fu = -g\dfrac{\partial h}{\partial y}
        - kv + \nu \left(\dfrac{\partial^2v}{\partial x^2} 
        + \dfrac{\partial^2v}{\partial y^2}\right).
\end{aligned}
$$
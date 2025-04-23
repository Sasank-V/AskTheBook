from manim import *


class ConvolutionalNeuralNetwork(Scene):
    def construct(self):
        # Title
        title = Text("Decoding the CNN: Layers of Deep Learning", font_size=36)
        self.play(Write(title))
        self.wait(1)

        # Introduction
        intro_text = Text("How do computers see images?", font_size=24).to_edge(UP)
        self.play(FadeIn(intro_text, shift=DOWN))

        # Show the CNN equation
        # equation = MathTex(r"\text{Image} \rightarrow \text{Feature Extraction} \rightarrow \text{Classification}")
        # self.play(Write(equation.next_to(intro_text, DOWN)))
        # self.wait(2)

        # Convolutional layer 1
        conv_title = Text("Convolutional Layers", font_size=24).to_edge(UP)
        self.play(Transform(intro_text, conv_title))

        # Convolutional kernel visualization
        kernel = Square(side_length=0.5, color=BLUE).shift(DOWN * 2 + LEFT * 3)
        image = Rectangle(height=4, width=6, fill_color=WHITE, fill_opacity=0.7).shift(
            UP
        )

        self.play(Create(image))
        self.play(FadeIn(kernel))

        # Convolution operation
        filter_text = Text("Filters (Kernels)", font_size=20).next_to(kernel, DOWN)
        equation2 = MathTex(r"\text{Filter} \ast \text{Input} = \text{Feature Map}")
        equation2.next_to(filter_text, DOWN)

        self.play(Write(filter_text), Write(equation2))

        feature_map = Rectangle(
            height=4, width=6, fill_color=YELLOW, fill_opacity=0.3
        ).shift(DOWN)
        feature_text = Text("Feature Maps", font_size=20).next_to(feature_map, DOWN)

        self.play(Create(feature_map), Write(feature_text))
        self.wait(2)

        # Pooling layer
        pooling_title = Text("Pooling Layers", font_size=24).to_edge(UP)
        self.play(Transform(intro_text, pooling_title))

        max_pool = Square(side_length=4, fill_color=RED).shift(DOWN)
        pool_text = Text("Max Pooling", font_size=20).next_to(max_pool, DOWN)

        # Highlight maximum value (pooling)
        highlight = SurroundingRectangle(max_pool, color=YELLOW)
        self.play(Create(highlight), Write(pool_text))

        # Fully connected layer
        fc_title = Text("Fully Connected Layer", font_size=24).to_edge(UP)
        self.play(Transform(intro_text, fc_title))

        fully_connected = Rectangle(height=2, width=4, fill_color=PURPLE).shift(
            DOWN * 2
        )
        output_text = Text("Output (Classification)", font_size=20).next_to(
            fully_connected, DOWN
        )

        equation3 = MathTex(
            r"\text{Flattened Feature Maps} \rightarrow \text{Weights} + \text{Activation}"
        )
        equation3.next_to(fully_connected, UP)

        self.play(Create(fully_connected), Write(output_text), Write(equation3))

        # Final CNN
        final_title = Text("The Complete CNN", font_size=36)
        self.play(Transform(title, final_title))

        # Show all components together
        complete_network = VGroup(
            image.copy().scale(0.8).shift(UP),
            kernel.copy().scale(0.6).shift(DOWN * 1 + LEFT * 3),
            feature_map.copy().scale(0.7).shift(DOWN),
            max_pool.copy().scale(0.5).shift(DOWN),
            fully_connected.copy().scale(0.8),
        )

        self.play(FadeIn(complete_network))
        self.wait(2)

        # Conclusion
        conclusion = Text(
            "CNN: A network that processes images and makes predictions", font_size=24
        ).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(3)


def render_scene():
    from manim import config, tempconfig

    with tempconfig(
        {
            "quality": "low_quality",
            "preview": True,
        }
    ):
        scene = ConvolutionalNeuralNetwork()
        scene.render()


# Call the function
render_scene()

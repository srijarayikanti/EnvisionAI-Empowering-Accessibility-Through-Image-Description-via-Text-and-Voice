from flask import Flask, render_template, redirect, request

import caption_it
#app = Flask(__name__)

app=Flask(__name__,template_folder='template')
@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def marks():
    if request.method == 'POST':
        f = request.files['userfile']

        path = "./static/{}".format(f.filename)
        f.save(path)
        audio_path ="../static/audio.mp3"
        caption = caption_it.caption_this_image(path)
        print("shashi : " + caption)
        result_dic ={
            'image' :path,
            'caption' :caption,
            'audiopath': audio_path
        }
    return render_template("index.html", result_c =result_dic)


if __name__ == '__main__':
    app.run(debug=True)

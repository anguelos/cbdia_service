import cbphocnet
import fargv
import string
import cherrypy
import json
import glob
from PIL import Image
import numpy as np
import time
from io import BytesIO
import base64

#cbphocnet.build_phoc_descriptor(["hello"], params.phoc_chars, params.phoc_levels)


class MockServer(object):
    def set_random_data(self, params):
        print("Creating random data ... ", end="")
        t = time.time()
        self.embeddings[:, :self.embedding_size//2] = np.random.rand(params.nb_embeddings, self.embedding_size//2)
        self.embeddings[:, self.embedding_size // 2:] = np.random.rand(params.nb_embeddings, self.embedding_size - self.embedding_size // 2)
        self.rects_ltrb[:, 0] = np.random.randint(0, params.page_width, params.nb_embeddings)
        self.rects_ltrb[:, 1] = np.random.randint(0, params.page_height, params.nb_embeddings)
        self.rects_ltrb[:, 2] = self.rects_ltrb[:, 0] + np.random.randint(params.min_word_width, params.max_word_width,
                                                                          params.nb_embeddings)
        self.rects_ltrb[:, 3] = self.rects_ltrb[:, 1] + np.random.randint(params.min_word_height,
                                                                          params.max_word_height, params.nb_embeddings)
        self.page_nums[:] = np.random.randint(0, params.pages_per_doc, params.nb_embeddings)
        self.doc_nums[:] = np.random.randint(0, self.document_names.shape[0], params.nb_embeddings)
        print(f"done! in {time.time()-t:5.3}")
        print("Embeedings",self.embeddings.shape)
        print("rects_ltrb", self.rects_ltrb.shape)
        print("page_nums", self.page_nums.shape)
        print("doc_nums", self.doc_nums.shape)

    def __init__(self, params):
        super().__init__()
        self.phoc_chars = params.phoc_chars
        self.phoc_levels = params.phoc_levels
        self.embedding_size = sum(self.phoc_levels) * len(self.phoc_chars)
        self.embeddings = np.empty([params.nb_embeddings, self.embedding_size], dtype=np.float32)
        self.rects_ltrb = np.empty([params.nb_embeddings, 4], dtype=np.short)
        self.page_nums = np.empty(params.nb_embeddings, dtype=np.short)
        self.doc_nums = np.empty(params.nb_embeddings, dtype=np.short)
        self.document_paths = np.array(glob.glob(params.document_glob))
        self.document_names = np.array([name.split("/")[-1].split("-")[0] for name in self.document_paths.tolist()])
        self.set_random_data(params)
        self.identity_idx = np.arange(0, params.nb_embeddings, dtype=np.long)

    def search(self, embedding, context=None):
        print("Searching indexes ... ", end="")
        t = time.time()
        embedding = embedding.astype("float32")
        if context is None:
            idx = np.argsort(((self.embeddings - embedding) ** 2).sum(axis=1))
        else:
            context_idx = self.doc_nums == -1
            for book_id in context:
                context_idx = context_idx|(self.doc_nums == (np.where(self.document_names == book_id)[0][0]))
            contextualised_embeddings = self.embeddings[context_idx, :]
            contextualised_idx = np.argsort(((contextualised_embeddings - embedding) ** 2).sum(axis=1))
            idx = self.identity_idx[context_idx][contextualised_idx] # todo(anguelos) check this out
        print(f"done! in {time.time() - t:5.3}")
        return idx

    def format_replies(self, idx):
        rectangles = [[self.document_names[self.doc_nums[n]],self.page_nums[n]]+self.rects_ltrb[n, :].tolist() for n in
                      idx.tolist()]
        rectangles = [[r[0], int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5])] for r in rectangles]
        return {"rectangles": rectangles}

    @cherrypy.expose
    def searchword(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        body = json.loads(rawbody)
        # do_something_with(body)
        print(repr(body))
        return f"Updated {body}."

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def qbs(self):
        query = cherrypy.request.json
        embedding = cbphocnet.build_phoc_descriptor([query["q_str"]], params.phoc_chars, params.phoc_levels)
        responce_idx = self.search(embedding)
        responces = self.format_replies(responce_idx)
        return responces

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def qbe(self):
        query = cherrypy.request.json
        img=Image.open(BytesIO(base64.b64decode(query["q_img"])))
        img.save("/tmp/query.png")
        embedding = np.random.rand(1,self.embedding_size)
        responce_idx = self.search(embedding)
        responces = self.format_replies(responce_idx)
        return responces

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def qbe_rotate(self):
        cherrypy.response.headers['Content-Type'] = 'image/png'
        query = cherrypy.request.json
        img=Image.open(BytesIO(base64.b64decode(query["q_img"])))
        img=img.rotate(45)
        byte_io = BytesIO()
        img.save(byte_io, 'PNG')
        return byte_io.getvalue()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def qbr(self):
        query = cherrypy.request.json
        embedding = np.random.rand(1, self.embedding_size)
        responce_idx = self.search(embedding)
        responces = self.format_replies(responce_idx)
        return responces

    @cherrypy.expose
    def index(self):
        return """
<html>
<script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.4.2/jquery.min.js"></script>
<script type='text/javascript'>
function UpdateSearchWord() {
    $.ajax({
      type: 'POST',
      url: "searchword",
      contentType: "application/json",
      processData: false,
      data: $('#searchwordbox').val(),
      success: function(data) {alert(data);},
      dataType: "application/json"
    });
}
function UpdateQBS() {
    $.ajax({
      type: 'POST',
      url: "qbs",
      contentType: "application/json",
      processData: false,
      data: $('#qbsbox').val(),
      success: function(data) {alert(data);},
      dataType: "application/json"
    });
}
function UpdateQBE() {
    $.ajax({
      type: 'POST',
      url: "qbe",
      contentType: "application/json",
      processData: false,
      data: $('#qbebox').val(),
      success: function(data) {alert(data);},
      dataType: "application/json"
    });
}
function DebugQBE() {
    $.ajax({
      type: 'POST',
      url: "qbe_rotate",
      contentType: "application/json",
      processData: false,
      data: $('#qbebox').val(),
      success: function(data) {alert(data);},
      dataType: "image/png"
    });
}


</script>
<body>
<input type='textbox' id='searchwordbox' value='{}' size='20' />
<input type='submit' value='SearchWord' onClick='UpdateSearchWord(); return false' />
<hr>
<input type='textbox' id='qbsbox' value='{"q_str":"hello"}' size='20' />
<input type='submit' value='Query By String' onClick='UpdateQBS(); return false' />
<hr>
<input type='textbox' id='qbebox' value='{"q_img":"R0lGODlhDwAPAKECAAAAzMzM/////wAAACwAAAAADwAPAAACIISPeQHsrZ5ModrLl
N48CXF8m2iQ3YmmKqVlRtW4MLwWACH+H09wdGltaXplZCBieSBVbGVhZCBTbWFydFNhdmVyIQAAOw=="}' size='20' />
<input type='submit' value='Query By Example' onClick='UpdateQBE(); return false' />
<input type='submit' value='Rotate Image' onClick='DebugQBE(); return false' />
</body>
</html>
"""

params = {
    "phoc_chars": string.ascii_lowercase+string.digits,
    "phoc_levels": [1, 2, 3, 4, 5],
    "nb_embeddings": 100000,
    "min_word_width": 50,
    "max_word_width": 150,
    "min_word_height": 50,
    "max_word_height": 150,
    "page_width": 2000,
    "page_height": 3000,
    "pages_per_doc": 10,
    "document_glob": "data/root/chronicles/*"
}

params, _ = fargv.fargv(params)

cherrypy.quickstart(MockServer(params))
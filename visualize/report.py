'''
viewprobe creates visualizations for a certain eval.
'''

import re
import numpy
# from scipy.misc import imread, imresize, imsave
from imageio import imread, imsave
from PIL import Image
import visualize.expdir as expdir
import visualize.bargraph as bargraph
import settings
import numpy as np
# unit,category,label,score

replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]', ''),
    (r'-[sc] ', ' '),
    (r'-[sc]\)', ')'),
    (r'_', ' '),
    ]]

def fix(s):
    for pattern, subst in replacements:
        s = re.sub(pattern, subst, s)
    return s


def upsample(features, shape):
    return np.array(Image.fromarray(features).resize(shape, resample=Image.BILINEAR))

def get_unit_acts(ufeat, uthresh, mask_shape, data_size):
    """
    Returns the activation of units
    """
    uidx = np.argwhere(ufeat.max((1, 2)) > uthresh).squeeze(1)
    ufeat = np.array([upsample(ufeat[i], mask_shape) for i in uidx])

    # Create full array
    uhitidx = np.zeros((data_size, *mask_shape), dtype=np.bool)

    # Change mask to bool based on threshold
    uhit_subset = ufeat > uthresh    
    uhitidx[uidx] = uhit_subset

    return uhitidx

def get_top_imgs_idx(ufeat, uthresh, mask_shape, data_size):
    uall_uhitidx = get_unit_acts(
        ufeat, uthresh, mask_shape, data_size
    )

    act_counts = uall_uhitidx.sum((1, 2))
    act_idxs = act_counts.argsort()[::-1]

    return act_idxs

def break_expl(expl):
    expl = re.sub("\(|\)", "", expl)
    expl = re.sub(r" AND NOT ", r" AND_NOT ", expl)
    # expl = re.sub(r"([A-Z]{3}) ", r"\1__", expl)
    # expl = expl.split("__")
    expl = re.split(" AND | OR | AND_NOT | NOT ", expl)
    # print()
    # print(expl)
    return expl

def breakup_tally_result(tally_result):
    new_tally_result = []
    for record in tally_result:
        cats = break_expl(record["category"])
        expls = break_expl(record["label"])
        
        for i, expl in enumerate(expls):
            new_tally_result.append({
                "unit": record["unit"],
                "category": cats[i],
                "label": expl,
                "detacc": record["detacc"]
            })
    
    return new_tally_result


def generate_html_summary(ds, layer, mask_shape, maxfeature=None, features=None, thresholds=None,
        imsize=None, imscale=72, tally_result=None,
        gridwidth=None, gap=3, limit=None, force=True, verbose=False):
    ed = expdir.ExperimentDirectory(settings.OUTPUT_FOLDER)
    print('Generating html summary %s' % ed.filename('html/%s.html' % expdir.fn_safe(layer)))

    if verbose:
        print('Sorting units by score.')
    if imsize is None:
        imsize = settings.IMG_SIZE

    # top = np.argsort(maxfeature, 0)[:-1 - settings.TOPN:-1, :].transpose()
    
    ed.ensure_dir('html','image')
    html = [html_prefix]
    rendered_order = []
    barfn = 'image/%s-bargraph.svg' % (
            expdir.fn_safe(layer))
    broken_tally_result = breakup_tally_result(tally_result)
    bargraph.bar_graph_svg(ed, layer,
                           tally_result=broken_tally_result,
                           rendered_order=rendered_order,
                           save=ed.filename('html/' + barfn))
    html.extend([
        '<div class="histogram">',
        '<img class="img-fluid" src="%s" title="Summary of %s %s">' % (
            barfn, ed.basename(), layer),
        '</div>'
        ])
    html.append('<div class="gridheader">')
    html.append('<div class="layerinfo">')
    # html.append('%d/%d units covering %d concepts with IoU &ge; %.2f' % (
    #     len([record for record in rendered_order
    #         if float(record['score']) >= settings.SCORE_THRESHOLD]),
    #     len(rendered_order),
    #     len(set(record['label'] for record in rendered_order
    #         if float(record['score']) >= settings.SCORE_THRESHOLD)),
    #     settings.SCORE_THRESHOLD))
    html.append('</div>')
    html.append(html_sortheader)
    html.append('</div>')

    if gridwidth is None:
        gridname = ''
        gridwidth = settings.TOPN
        gridheight = 1
    else:
        gridname = '-%d' % gridwidth
        gridheight = (settings.TOPN + gridwidth - 1) // gridwidth

    html.append('<div class="unitgrid"') # Leave off > to eat spaces
    if limit is not None:
        tally_result = tally_result[:limit]
    for i, record in enumerate(sorted(tally_result, key=lambda record: -float(record['detacc']))):
        record['score-order'] = i
    
    for i, record in enumerate(sorted(tally_result, key=lambda record: -float(record['iou']))):
        record['iou-order'] = i
    
    for label_order, record in enumerate(
        sorted(
            tally_result,
            key=lambda record: re.sub("[\(\)]", "", record["label"])
        )
    ):
        unit = int(record['unit']) #- 1 # zero-based unit indexing
        
        imfn = 'image/%s%s-%04d.jpg' % (
                expdir.fn_safe(layer), gridname, unit)
        if force or not ed.has('html/%s' % imfn):
            if verbose:
                print('Visualizing %s unit %d' % (layer, unit))
            # Generate the top-patch image
            tiled = numpy.full(
                ((imsize + gap) * gridheight - gap,
                 (imsize + gap) * gridwidth - gap, 3), 255, dtype='uint8')
            
            # for x, index in enumerate(top[unit]):
            img_idx = get_top_imgs_idx(features[:,unit], thresholds[unit], mask_shape, features.shape[0])
            
            for x in range(settings.TOPN):
                index = img_idx[x]
                row = x // gridwidth
                col = x % gridwidth
                image = imread(ds.filename(index))
                # mask = imresize(features[index][unit], image.shape[:2], mode='F')
                mask = numpy.array(Image.fromarray(features[index][unit], mode='F').resize(image.shape[:2]))
                mask = mask > thresholds[unit]
                vis = (mask[:, :, numpy.newaxis] * 0.8 + 0.2) * image
                if vis.shape[:2] != (imsize, imsize):
                    vis = numpy.array(Image.fromarray(vis).resize((imsize, imsize)))
                tiled[row*(imsize+gap):row*(imsize+gap)+imsize,
                      col*(imsize+gap):col*(imsize+gap)+imsize,:] = vis
            imsave(ed.filename('html/' + imfn), tiled)
        # Generate the wrapper HTML
        # graytext = ' lowscore' if float(record['detacc']) < settings.SCORE_THRESHOLD else ''
        graytext = ''
        html.append('><div class="unit%s" data-order="%d %d %d %d">' %
                (graytext, label_order, unit + 1, record['score-order'], record['iou-order']))
        html.append('<div class="unitlabel">%s</div>' % fix(record['label']))
        html.append('<div class="info">' +
            '<span class="layername">%s</span> ' % layer +
            '<span class="category">(%s)</span> ' % record['category'] +
            '<br><span class="unitnum">Unit %d DetAcc %.2f IoU %.2f</span> ' % (unit, float(record['detacc']), float(record['iou'])) +
            
            # '<span class="iou">DetAcc %.2f <br>IoU %.2f</span>' % (float(record['detacc']), float(record['iou'])) +
            '</div>')
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>' %
            (imfn, imscale))
        html.append('</div') # Leave off > to eat spaces
    html.append('></div>')
    html.extend([html_suffix]);
    with open(ed.filename('html/%s.html' % expdir.fn_safe(layer)), 'w') as f:
        f.write('\n'.join(html))

html_prefix = '''
<!doctype html>
<html>
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css">
<script src="https://code.jquery.com/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js" integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn" crossorigin="anonymous"></script>
<style>
.unitviz, .unitviz .modal-header, .unitviz .modal-body, .unitviz .modal-footer {
  font-family: Arial;
  font-size: 15px;
}
.unitgrid {
  text-align: center;
  border-spacing: 5px;
  border-collapse: separate;
}
.unitgrid .info {
  text-align: left;
}
.unitgrid .layername {
  display: none;
}
.unitlabel {
  font-weight: bold;
  font-size: 110%;
  text-align: left;
  line-height: 1;
}
.lowscore .unitlabel {
   color: silver;
}
.thumbcrop {
  overflow: hidden;
  width: 288px;
  height: 72px;
}
.unit {
  display: inline-block;
  background: white;
  padding: 3px;
  margin: 2px;
  box-shadow: 0 5px 12px grey;
}
.iou {
  display: inline-block;
  float: right;
}
.modal .big-modal {
  width:auto;
  max-width:90%;
  max-height:80%;
}
.modal-title {
  display: inline-block;
}
.footer-caption {
  float: left;
  width: 100%;
}
.histogram {
  text-align: center;
  margin-top: 3px;
}
.img-wrapper {
  text-align: center;
}
.big-modal img {
  max-height: 60vh;
}
.img-scroller {
  overflow-x: scroll;
}
.img-scroller .img-fluid {
  max-width: initial;
}
.gridheader {
  font-size: 12px;
  margin-bottom: 10px;
  margin-left: 30px;
  margin-right: 30px;
}
.gridheader:after {
  content: '';
  display: table;
  clear: both;
}
.sortheader {
  float: right;
  cursor: default;
}
.layerinfo {
  float: left;
}
.sortby {
  text-decoration: underline;
  cursor: pointer;
}
.sortby.currentsort {
  text-decoration: none;
  font-weight: bold;
  cursor: default;
}
</style>
</head>
<body class="unitviz">
<div class="container-fluid">
'''

html_sortheader = '''
<div class="sortheader">
sort by
<span class="sortby currentsort" data-index="0">Label</span>
<span class="sortby" data-index="1">Unit</span>
<span class="sortby" data-index="2">DetAcc</span>
<span class="sortby" data-index="3">IoU</span>
</div>
'''

html_suffix = '''
</div>
<div class="modal" id="lightbox">
  <div class="modal-dialog big-modal" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title"></h5>
        <button type="button" class="close"
             data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <div class="img-wrapper img-scroller">
          <img class="fullsize img-fluid">
        </div>
      </div>
      <div class="modal-footer">
        <div class="footer-caption">
        </div>
      </div>
    </div>
  </div>
</div>
<script>
$('img:not([data-nothumb])[src]').wrap(function() {
  var result = $('<a data-toggle="lightbox">')
  result.attr('href', $(this).attr('src'));
  var caption = $(this).closest('figure').find('figcaption').text();
  if (!caption && $(this).closest('.citation').length) {
    caption = $(this).closest('.citation').text();
  }
  if (caption) {
    result.attr('data-footer', caption);
  }
  var title = $(this).attr('title');
  if (!title) {
    title = $(this).closest('td').find('.unit,.score').map(function() {
      return $(this).text(); }).toArray().join('; ');
  }
  if (title) {
    result.attr('data-title', title);
  }
  return result;
});
$(document).on('click', '[data-toggle=lightbox]', function(event) {
    $('#lightbox img').attr('src', $(this).attr('href'));
    $('#lightbox .modal-title').text($(this).data('title') ||
       $(this).closest('.unit').find('.unitlabel').text());
    $('#lightbox .footer-caption').text($(this).data('footer') ||
       $(this).closest('.unit').find('.info').text());
    event.preventDefault();
    $('#lightbox').modal();
    $('#lightbox img').closest('div').scrollLeft(0);
});
$(document).on('keydown', function(event) {
    $('#lightbox').modal('hide');
});
$(document).on('click', '.sortby', function(event) {
    var sortindex = +$(this).data('index');
    sortBy(sortindex);
    $('.sortby').removeClass('currentsort');
    $(this).addClass('currentsort');
});
function sortBy(index) {
  $('.unitgrid').find('.unit').sort(function (a, b) {
     return +$(a).eq(0).data('order').split(' ')[index] -
            +$(b).eq(0).data('order').split(' ')[index];
  }).appendTo('.unitgrid');
}
</script>
</body>
</html>
'''

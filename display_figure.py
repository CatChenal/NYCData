"""
Wrapper to place an image inside an html <figure tag>.
Allows flexible tag-styling by passing style dicts.

# Example:
style_dict = {'div': {'width': 75},
              'figure': {'display':'inline-block', 'text-align':'center'},
              'image': {'display':'block', 'width': 700, 'height':500},
              'caption': {'color':'teal','font-weight':'bold', 'font-family': 'Arial, Helvetica, sans-serif'} }
caption_dict = {'number': 1, 'caption': 'Picture from wikipedia <a href="https://en.wikipedia.org/wiki/Sancerre">Sancerre</a>'}

fname = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Sancerre_france.jpg/405px-Sancerre_france.jpg'
display_figure(fname, style_dict, caption_dict, img_title='pic from wikipedia')

# To change something in the styles/caption, e.g. due to:
# . cell reordering
# . adjustment to the width
# then you only need to update the dict values and rerun:

#style_dict['div'].update([('width',100)])
#style_dict['figure'].update([('text-align','left')])
#caption_dict.update([('number',3)])
#display_figure(fname, style_dict, caption_dict, img_title='pic from wikipedia')

"""
from IPython.display import HTML


def display_figure(img_path_or_url, style_dict, caption_dict,
                   img_title='', return_html=True):
    """
    style_dict example:
       {'figure': {'display':'inline-block', 'text-align':'center'},
        'image': {'display':'block', 'width': 500, 'height':500},
        'caption': {'color':'teal','font-weight':'bold', 'font-family': 'Arial, Helvetica, sans-serif'}
        }
    caption_dict example:
       {'number': 1, 'caption': 'Picture/plot shows...'}
       
    img_title: this is the 'on hover' text.
    return_html: if True, the html string is returned
    """
    if (not isinstance(style_dict, dict) or not isinstance(caption_dict, dict)):
        print('Please provide dictionnaries to set the style and caption details.')
        return
    
    def get_style_strings(style_dict):
        fig_style = ''
        img_style = ''
        cap_style = ''
        
        figure_style = style_dict.get('figure')
        if figure_style is not None:
            for k, v in figure_style.items():
                fig_style += f'{k}:{v};'

        image_style = style_dict.get('image')
        if image_style is not None:
            for k, v in image_style.items():
                img_style += f'{k}:{v};'  

        caption_style = style_dict.get('caption')
        if caption_style is not None:
            for k, v in caption_style.items():
                cap_style += f'{k}:{v};'

        return fig_style, img_style, cap_style
    
    fig_style, img_style, cap_style = get_style_strings(style_dict)
        
    cap_number = caption_dict.get('number')
    caption_text = caption_dict.get('caption')
        
    html = f"""<figure style="{fig_style}">
      <img src="{img_path_or_url}" 
         alt="x"
         style="{img_style}"
         title="{img_title}"
      >
      <figcaption style="{cap_style}">
        Figure {cap_number} - {caption_text}
      </figcaption></figure>
    """
    if return_html:
        return html
    else:
        return HTML(html)

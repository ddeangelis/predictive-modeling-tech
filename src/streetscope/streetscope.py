'''
    File name: streetscope.py
    Author: Tyche Analytics Co.
'''
import subprocess as sub
import pyqtree
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from tqdm import *
import os
import usaddress
from math import sqrt
from utils3 import choose2, mean
from zcta import zip_feature
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def geocode_dep(address, spacer="+", verbose=False):
    search_template = "curl --globoff localhost/search/%s?format=json"
    filled_template = search_template % address.replace(" ", spacer)
    if verbose:
        print("template:", filled_template)
    p = sub.Popen(filled_template.split(), stdout=sub.PIPE,stderr=sub.PIPE)
    raw_output, errors = p.communicate()
    if verbose:
        print(raw_output)
        print(errors)
    try:
        all_output = eval(raw_output)
    except:
        return None
    if len(all_output) != 1:
        print("WARNING:", address, "results:", len(all_output))
        print(all_output)
        if len(all_output) == 0:
            return None
    output = all_output[0]
    lat, lon = float(output['lat']), float(output['lon'])
    return lat, lon
    #return output

# http://nominatim.openstreetmap.org/search?q=[Tankstelle]&format=xml&limit=50&viewbox=7.98435,49.40889,8.95440,48.77371&bounded=1
# west north east south
def _geocode(query, **kwargs):
    cmd_template = "curl --globoff"
    url_template = "localhost/search/"
    query_template = query.replace(" ", "%20")
    url = url_template + query_template
    if kwargs:
        kwarg_template = "&".join(k + "=" + str(v) for k,v in kwargs.items())
        url += "?" + kwarg_template
    final_template = cmd_template + " " + url
    p = sub.Popen(final_template.split(), stdout=sub.PIPE,stderr=sub.PIPE)
    raw_output, errors = p.communicate()
    print("raw output:", raw_output)

    try:
        output = eval(raw_output)
    except SyntaxError:
        print("encountered SyntaxError on output eval, treating as html")
        soup = BeautifulSoup(raw_output)
        text = soup.get_text()
        print("bs text:", text)
        idx = text.index('[') # skip past warnings, look for start of json dict
        try:
            output = eval(text[idx:])
            print("succeeded on html parsing")
        except:
            print("failed html parsing")
            output = None
        
    return output

def parse_address(add, remove_commas=True):
    """given address, return parsed defaultdict"""
    if remove_commas:
        add = add.replace(",", " ")
    return defaultdict(lambda :"",
                       {y:x for (x, y) in usaddress.parse(add)})
    
def geocode(add, verbose=False, filter_results=True):
    for aggression_level in range(-1, 2+1):
        normed_address = normalize_address(add, aggression=aggression_level)
        if verbose:
            print("attempting to geocode:", normed_address, "aggression:", aggression_level)
        ans =  _geocode(normed_address, format='json')
        if ans:
            if verbose:
                print("succeeded on aggression level:", aggression_level)
            return (filter_geocoding(add, ans) if filter_results else ans)
    else:
        if verbose:
            print("failed on:", normed_address, "aggression level:", aggression_level)
            print("recoursing to zip data")
        parsed = parse_address(add)
        if 'ZipCode' in parsed:
            z = parsed['ZipCode']
            lat, lon = zip_feature(z, 'intptlat'), zip_feature(z, 'intptlon')
            return [{'lat':lat, 'lon':lon}]
    if verbose:
        print("failed on:", add)
    return None
    
def clean_address(address):
    subs = [(r"-[0-9]+", r''),
            (r"hwy", "highway"),
            (r"ste.? [0-9]+", "")]
    out = address
    for search, replace in subs:
        out = re.sub(search, replace, out, flags=re.IGNORECASE)
        print(out)
    return out
    
def lookup_amenities(address):
    lat_lon = geocode(address, spacer="+")
    if lat_lon is None:
        lat_lon = geocode(address, spacer="%20")
    if lat_lon is None:
        return None
    else:
        lat, lon = lat_lon
    print("lat, lon:", lat, lon)
    return amenities_from_lat_lon(lat, lon, delta=0.01)

def amenities_from_lat_lon(lat, lon, delta=0.01):
    amenities = []
    viewbox=",".join(map(str, [lon - delta, lat + delta, lon + delta, lat - delta]))
    print("viewbox:", viewbox)
    for atype in amenity_types:
        query = "[%s]" % atype.replace("_", " ")
        output = geocode(query, format="json", limit=50,
                                  viewbox=viewbox, bounded=1)
        amenities.append(output)
    return sum(amenities, [])
    
def foo(radius, lat, lon):
    query_template = "curl localhost/query/%s?format=json"
    query = 'node["amenity"](around:{}, {}, {});out;'.format(radius, lat, lon)
    filled_template = query_template % address.replace(" ", "%20")
    p = sub.Popen(filled_template.split(), stdout=sub.PIPE,stderr=sub.PIPE)
    raw_output, errors = p.communicate()

def bar(address):
    search_template = "curl --globoff localhost/search/%s?format=json"
    filled_template = search_template % address#.replace(" ", "%20")
    p = sub.Popen(filled_template.split(), stdout=sub.PIPE,stderr=sub.PIPE)
    raw_output, errors = p.communicate()
    print(raw_output)
    output = eval(raw_output)[0]
    lat, lon = output['lat'], output['lon']
    return lat, lon

def get_amenity_types():
    with open("<PATH>/amenity_types.csv") as f:
        fields = [line.split(",") for line in f.readlines()]
    amenities = [field[2] for field in fields if field[1] == 'amenity']
    return sorted(set(amenities))

amenity_types = get_amenity_types()

import osmium
import sys

class NamesHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.num_amenities = 0
        self.results = []
        self.tried = 0
        self.ways = []
        
    def get_amenities(self, n):
        if n.tags.get('amenity'):
            self.num_amenities += 1
            self.results.append((n.location, n.tags['amenity'],
                                 n.tags.get('name'),
                                 n.tags.get('display_name'),
                                 n.tags.get('osm_id')))
            #self.results.append((n.location, n.tags))
            if self.num_amenities % 100 == 0:
                # print(n.tags)
                print(n.location, n.tags['amenity'], self.num_amenities)
        self.tried += 1
        if self.tried % 1000000 == 0:
            print("tried:", self.tried)

    def node(self, n):
        pass
        #self.get_amenities(n)

    def way(self, w):
        if w.tags.get('amenity'):
            self.ways.append(w)
            print(len(self.ways))
            if len(self.ways) > 10:
                raise Exception()
            self.get_amenities(w)

def load_amenities():
    nh = NamesHandler()
    #nh.apply_file('/srv/public_data/na-amenities.osm.bz2')
    nh.apply_file('<PATH>/north-america-latest.osm.bz2')
    return nh.results

def build_qtree_ref(results):
    min_lat = min(x[0].lat for x in results)
    max_lat = max(x[0].lat for x in results)
    min_lon = min(x[0].lon for x in results)
    max_lon = max(x[0].lon for x in results)
    spindex = pyqtree.Index(bbox=(min_lon, min_lat, max_lon, max_lat))
    for loc, atype, name in tqdm(results):
        lat, lon = loc.lat, loc.lon
        bbox = (lon, lat, lon, lat)
        spindex.insert(atype, bbox)
    return spindex

def build_qtree():
    with open("<PATH>/amenity_coords.csv") as f:
        xs = [(float(lat), float(lon), atype, name) for (lat, lon, atype, name)
              in [line.strip().split(",") for line in f.readlines()[1:]
                  if len(line.split(",")) == 4]]
    min_lat = min(x[0] for x in xs)
    max_lat = max(x[0] for x in xs)
    min_lon = min(x[1] for x in xs)
    max_lon = max(x[1] for x in xs)
    spindex = pyqtree.Index(bbox=(min_lon, min_lat, max_lon, max_lat))
    for lat, lon, atype, name in tqdm(xs):
        bbox = (lon, lat, lon, lat)
        spindex.insert((atype, name), bbox)
    return spindex

def intersect_ref(bbox, coords):
    (min_lon, min_lat, max_lon, max_lat) = bbox
    ans = []
    for lat, lon, atype, name in tqdm(coords):
        if (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon):
            ans.append((lat, lon, atype, name))
    return ans

def manipulate_xml():
    t = time.time()
    tree = ET.parse("<PATH>/na-amenities2.osm")
    t2 = time.time()
    print("parsing took:", (t2-t), "seconds")
    root = tree.getroot()
    amenities = []
    lat_lon_dict = {}
    for i, x in enumerate(root):
        if x.tag == 'node':
            id_ = x.attrib['id']
            lat = x.attrib['lat']
            lon = x.attrib['lon']
            lat_lon_dict[id_] = (float(lat), float(lon))
        if any('k' in child.attrib and 'amenity' == child.attrib['k'] for child in x):
            amenities.append(x)
            if len(amenities) % 10000 == 0:
                print(i, len(amenities), len(lat_lon_dict))
            return lat, lon, amenity, name
    def extract_coords(n):
        name = None
        if n.tag == 'node':
            lat = float(n.attrib['lat'])
            lon = float(n.attrib['lon'])
            for c in n:
                if c.attrib['k'] == "amenity":
                    amenity = c.attrib['v']
                elif c.attrib['k'] == "name":
                    name = c.attrib['v']
            return lat, lon, amenity, name
        elif n.tag == 'way':
            ids = []
            ids = [c.attrib['ref'] for c in n if 'ref' in c.attrib]
            for c in n:
                if 'ref' in c.attrib:
                    ids.append(c.attrib['ref'])
                elif c.attrib['k'] == 'amenity':
                    amenity = c.attrib['v']
                elif c.attrib['k'] == 'name':
                    name = c.attrib['v']
            lat_lons = [lat_lon_dict[i] for i in ids]
            lat, lon = Polygon(lat_lons).centroid.coords[0]
            return lat, lon, amenity, name

    coords = [extract_coords(x) for x in tqdm(amenities)]
    with open("amenity_coords.csv", 'w') as f:
        f.write("lat, lon, amenity_type, name\n")
        for line in coords:
            lat, lon, atype, name = map(str, line)
            name = name.replace(",", " ").replace("\n", " ").strip()
            if not "," in atype: # a few are miscoded
                f.write(",".join((lat, lon, atype, name)) + "\n")

def census_geocoding(ids, raw_addresses, cities, states, zips, address_dir='.'):
    """https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.pdf"""
    addresses = []
    print(list(map(len, [ids, raw_addresses, cities, states, zips])))
    for _id, add, city, state, z in tqdm(zip(ids, raw_addresses, cities, states, zips)):
        if "-" in add:
            add = add[:add.index("-")]
        for char in string.punctuation:
            add = add.replace(char, " ")
        addresses.append(", ".join([str(_id), add, city, state, z]))
    lines = (sorted(set(addresses)))
    these_lines = []
    fnames = []
    for i, line in enumerate(lines):
        these_lines.append(line)
        if len(these_lines) == 1000:
            fname = "addresses_" + str(i) + ".csv"
            print("fname:", fname)
            fnames.append(fname)
            with open(fname, 'w') as f:
                f.write("\n".join(these_lines))
            these_lines = []
    template = "curl --form addressFile=@%s --form benchmark=9 https://geocoding.geo.census.gov/geocoder/locations/addressbatch --output " + os.path.join(address_dir, "%s_result.csv")
    for fname in fnames:
        cmd = template % (fname, fname)
        print("running command:", cmd)
        p = sub.Popen(cmd.split(), stdout=sub.PIPE,stderr=sub.PIPE)
        raw_output, errors = p.communicate()
        print(raw_output)
        print(errors)

class StreetScope():
    def __init__(self, address_dir='.', qtree=None):
        if qtree is None:
            print("building qtree")
            self.qtree = build_qtree()
        else:
            self.qtree = qtree
        self.address_dir = address_dir
        
    def lookup_address(self, address, delta=0.01, verbose=False):
            lat_lon = geocode(address, spacer="+")
            if lat_lon is None:
                lat_lon = geocode(address, spacer="%20")
            if lat_lon is None:
                if verbose:
                    print("couldn't reverse geocode:", address)
                return None
            else:
                if verbose:
                    print("found lat lon:", lat_lon)
                lat, lon = lat_lon
                return self.lookup_lat_lon(lat, lon, delta=delta)
            
    def lookup_lat_lon(self, lat, lon, delta=0.01, verbose=False):
        bbox = (lon - delta, lat -delta, lon + delta, lat + delta)
        results = self.qtree.intersect(bbox)
        return Counter([atype for (atype, name) in results])

    def bulk_amenities(self, ids, streets, cities, states, zips, bb=0.01):
        """master method:
        ss = StreetScope()
        amenity_X = ss.bulk_amenities(df.index, df.street, df.city, df.state, df.zip)

        Task of bulk amenities is to perform transformations:
        1) address -> (lat, lon)
        2) (lat, lon) -> amenity list
        """
        lat_lons = self.geocode(ids, streets, cities, states, zips)
        amenity_X = self.get_amenities(lat_lons)

                
def analyze_addresses(address_dirname):
    lat_lons = {}
    addresses = {}
    bad_files = []
    for fname in tqdm(os.listdir(address_dirname)):
        print(fname)
        if not 'result' in fname:
            with open(os.path.join(address_dirname, fname)) as f:
                lines = f.readlines()
            for line in lines:
                b = line.index(',')
                uid = line[:b]
                add = line[b+1:]
                addresses[uid] = add.strip()
            continue
        elif 'result' in fname:
            try:
                with open(os.path.join(address_dirname, fname)) as f:
                    lines = f.readlines()
            except:
                print("couldn't read:", fname)
                continue
        for line in lines:
            try:
                fields = line.split(",")
                uid = eval(fields[0])
                if "No_Match" in line:
                    match = False
                elif "Match" in line:
                    match = True
                else:
                    #print("WARNING: couldn't find match:", line)
                    pass
                if "Non_Exact" in line:
                    exact = False
                elif "Exact" in line:
                    exact = True
                else:
                    if match is False:
                        exact = False
                    else:
                        #print("Match true, but couldn't find exact?", line)
                        exact = False
                regexp = r"-?[0-9]+\.[0-9]+"
                matches = re.findall(regexp, line)
                if matches:
                    lon, lat = map(float, matches)
                else:
                    lon, lat = None, None
                lat_lons[uid] = (match, exact, lat, lon)
            except:
                print("WARNING: something failed with", line)
    return lat_lons, addresses


from multiprocessing.dummy import Pool as ThreadPool
def geocode_lat_lons(lat_lons):
    pool = ThreadPool(4)
    def f(lat_lon):
        lat, lon = lat_lon
        if lat and lon:
            a = amenities_from_lat_lon(lat, lon)
            print(lat, lon, len(a))
            return a
        else:
            print(lat, lon, None)
            return None
    lls = [(v[2], v[3]) for v in lat_lons.values()]
    amenities = pool.map(f, tqdm(lls))
    return {ll:a for ll,a in zip(lls, amenities)}
            
def address_analysis(address_dirname):
    streetscope = StreetScope()
    lat_lons, addresses = analyze_addresses(address_dirname)
    uids = list(lat_lons.keys())
    amenities, valid_uids = [], []
    for uid in tqdm(uids):
        match, exact, lat, lon = lat_lons[uid]
        if not (lat, lon) == (None, None):
            a = streetscope.lookup_lat_lon(lat, lon)
            amenities.append(a)
            valid_uids.append(uid)
    amenity_types = defaultdict(int)
    for a in amenities:
        if a is None:
            continue
        for t in a:
            amenity_types[t] += 1
    amenity_types = [k for k, v in
                     sorted(amenity_types.items(),
                            key = lambda kv:kv[1],
                            reverse=True)[:100]]
    
    a_df = pd.DataFrame([[c[a] for a in amenity_types] for c in tqdm(amenities)])
    a_df.columns = amenity_types
    a_df.index = valid_uids
    return a_df

def normalize_address(add, return_sep=' ', aggression=0, verbose=False):
    if aggression == -1:
        return add
    elif aggression == 3:
        pass
    add = add.replace(",", " ")
    parsed = {y:x for (x, y) in usaddress.parse(add)}
    if verbose:
        print(parsed)
    parsed = defaultdict(lambda :"", parsed)
    parsed['comma'] = ","
    return_fields = ["AddressNumber" * (aggression <= 0),
                     'StreetNamePreDirectional' * (aggression <= 1),
                     "StreetName" * (aggression <= 2),
                     "StreetNamePostType" * (aggression <=2), 
                     "comma" * (aggression <=2),
                     "PlaceName" *(aggression <=2),
                     "StateName" *(aggression <=2),
                     "ZipCode"]
    return return_sep.join([parsed[field] for field in return_fields])
    
def internal_distance(results):
    """when multiple results are returned, calculate average distance between results"""
    L = len(results)
    if L == 1:
        return 0
    lat_lons = []
    for d in results:
        lat_lons.append((float(d['lat']), float(d['lon'])))
    return mean((sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2))
                for (lat1, lon1), (lat2, lon2) in choose2(lat_lons))

def filter_geocoding(add, ans):
    """filter geocoding,  returning average of all lat lon points in same zip as add"""
    if len(ans) == 1:
        d = ans[0]
        return (float(d['lat']), float(d['lon']))
    else:
        parsed_add = parse_address(add)
        lats, lons = [], []
        for d in ans:
            disp_name = d['display_name'].replace("United States of America", "")
            parsed_add_p = parse_address(disp_name)
            print(d['display_name'], parsed_add_p, parsed_add_p['ZipCode'], parsed_add['ZipCode'])
            if parsed_add_p['ZipCode'] == parsed_add['ZipCode']:
                lats.append(float(d['lat']))
                lons.append(float(d['lat']))
        return (mean(lats), mean(lons))
            
def make_amenity_df(amenities, dims=100):    
    amenity_types = defaultdict(int)
    for a in amenities:
        if a is None:
            continue
        for t in a:
            amenity_types[t] += 1
    amenity_types = [k for k, v in
                     sorted(amenity_types.items(),
                            key = lambda kv:kv[1],
                            reverse=True)[:dims]]

    a_df = pd.DataFrame([[c[a] for a in amenity_types] for c in tqdm(amenities)])
    a_df.columns = amenity_types
    return a_df

class StreetScopeTransformer(BaseEstimator, TransformerMixin):
    """Class for using streetscope within sklearn"""
    def __init__(self, delta=0.1, amenity_dims=100, qtree=None):
        print("foo")
        print("initializing streetscope transformer")
        self.qtree = qtree
        self.ss = StreetScope(qtree=self.qtree)
        self.amenity_dims = amenity_dims
        self.delta = delta
        self.amenity_types = None
        print("initialized streetscope transformer")

    def fit(self, lat_lons, ys):
        print("received lat lons with shape:", lat_lons.shape)
        #return lat_lons
        amenities = []
        for i, (lat, lon) in tqdm(lat_lons.iterrows(), total=len(lat_lons)):
            a = self.ss.lookup_lat_lon(lat, lon, delta=self.delta)
            amenities.append(a)
        amenity_types = defaultdict(int)
        for a in amenities:
            if a is None:
                continue
            for t in a:
                amenity_types[t] += 1
        amenity_types = [k for k, v in
                         sorted(amenity_types.items(),
                                key = lambda kv:kv[1],
                                reverse=True)[:self.amenity_dims]]
        print("amenity types:", amenity_types)
        self.amenity_types = amenity_types
        return self

    def transform(self, lat_lons):
        print("received lat lons with shape:", lat_lons.shape)
        amenities = []
        print("iterating")
        for i, (lat, lon) in tqdm(lat_lons.iterrows(), total=len(lat_lons)):
            a = self.ss.lookup_lat_lon(lat, lon, delta=self.delta)
            amenities.append(a)
        print("amenity len:", len(amenities))
        a_df = pd.DataFrame([[c[a] for a in self.amenity_types] for c in tqdm(amenities)])
        a_df.columns = self.amenity_types
        print(a_df.shape)
        return a_df

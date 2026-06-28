#!/usr/bin/env ruby

require 'date'
require 'nokogiri'
require 'yaml'

ALLOWED_TOP_CATEGORIES = ['공부 기록', '프로젝트', '취준 기록', '일상'].freeze

def assert(condition, message)
  raise message unless condition
end

Dir['_posts/**/*.md'].each do |path|
  source = File.read(path)
  front_matter = source[/\A---\s*\n(.*?)\n---\s*\n/m, 1]
  data = YAML.safe_load(front_matter, permitted_classes: [Date, Time], aliases: true)
  next if data['hidden'] == true

  top_category = Array(data['categories']).first
  assert(ALLOWED_TOP_CATEGORIES.include?(top_category), "#{path} has invalid top category #{top_category.inspect}")
end

home = Nokogiri::HTML(File.read('_site/index.html'))

assert(home.at_css('#panel-wrapper').nil?, 'home right panel must not render')
assert(home.at_css('#home-profile'), 'home profile is missing')
assert(home.css('#home-recent-posts .home-post-card').size == 3, 'home must show three recent posts')
assert(home.css('#home-projects .home-project-card').size <= 4, 'home must show at most four projects')
assert(home.css('#sidebar .sidebar-category-group').size == 4, 'sidebar must show four category groups')
assert(home.at_css('#topbar .home-identity'), 'home identity is missing from topbar')
assert(home.at_css('#search-input'), 'search input is missing')

puts 'home structure checks passed'
